import torch.nn as nn
import torch

class MaxGramExtractor(nn.Module):
    def __init__(self, question_word_list_length=5747, hidden_dim=512):
        super().__init__()
        # self.densify_oneHot = nn.Embedding(question_word_list_length, hidden_dim)
        self.densify_oneHot = nn.Linear(question_word_list_length, hidden_dim)
        self.threeGram = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.twoGram = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=2, padding=1)
        self.oneGram = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)

    def forward(self, oneHot_q):
        # print(oneHot_q.dtype, oneHot_q.cuda().dtype, oneHot_q.long().dtype)
        # latent_w = oneHot_q# self.densify_oneHot(oneHot_q.long())
        latent_w = self.densify_oneHot(oneHot_q)
        # shape is (batch x q_len x dim)
        latent_w = latent_w.permute(0, 2, 1)

        gram_3 = torch.tanh(self.threeGram(latent_w))
        gram_2 = torch.tanh(self.twoGram(latent_w)[:, :, :-1])
        gram_1 = torch.tanh(self.oneGram(latent_w))

        max_gram = torch.maximum(gram_3, gram_2)
        max_gram = torch.maximum(max_gram, gram_1)

        return latent_w.permute(0, 2, 1), max_gram.permute(0, 2, 1)
    

class HierarchicalQEncoding(nn.Module):
    def __init__(self, question_word_list_length=5747, hidden_dim=512):
        super().__init__()
        self.max_gram_extract = MaxGramExtractor(question_word_list_length, hidden_dim)
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)

    def forward(self, oneHot_q):#, q_len):
        latent_w, max_gram = self.max_gram_extract(oneHot_q)
        # packed_seq = nn.utils.rnn.pack_padded_sequence(max_gram, q_len, batch_first=True)
        s_out = self.lstm(max_gram)[0]
        # s_out = nn.utils.rnn.pad_packed_sequence(p_out, batch_first=True, total_length=max_gram.shape[1])

        return latent_w, max_gram, s_out
    
    
class AlternatingAttention(nn.Module):
    def __init__(self, hidden_dim=512):
        super().__init__() 
        self.g_linear = nn.Linear(hidden_dim, hidden_dim)
        self.x_linear = nn.Linear(hidden_dim, hidden_dim)
        self.hx_linear = nn.Linear(hidden_dim, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.hidden_dim = hidden_dim

    def attentionOperator(self, input_features, guidance):
        a_x = torch.mul(input_features, guidance)
        # print("mul ", a_x.shape)
        a_x = a_x.sum(dim=1)
        # print("sum ", a_x.shape)
        return a_x.squeeze()

    def crossAttention(self, input_features, guidance):
        # print("input_features", input_features.shape, guidance.shape)
        left_term = self.x_linear(input_features)
        right_term = self.g_linear(guidance).unsqueeze(1)
        # print("left right", left_term.shape, right_term.shape)
        H = torch.tanh(left_term + right_term)
        a_x = self.hx_linear(H).squeeze(-1)
        # # print("attention ", a_x.shape)
        x_i = self.softmax(a_x).unsqueeze(-1)
        # print("softmax ", x_i.shape)
        out_vector = self.attentionOperator(input_features, x_i)

        return out_vector
    
    def forward(self, q_vect, img_vect):
        s_hat = self.crossAttention(q_vect, torch.zeros(q_vect.shape[0], self.hidden_dim).cuda())
        v_hat = self.crossAttention(img_vect, s_hat)
        q_hat = self.crossAttention(q_vect, v_hat)

        return v_hat, q_hat

class CoattentionNet(nn.Module):
    """
    Predicts an answer to a question about an image using the Hierarchical Question-Image Co-Attention
    for Visual Question Answering (Lu et al, 2017) paper.
    """
    def __init__(self, q_list_len=5747, hid_d=512, ans_list_len=5217):
        super().__init__()
        ############ 3.3 TODO
        self.hierQEnc = HierarchicalQEncoding(question_word_list_length=q_list_len, hidden_dim=hid_d)
        self.w_attention = AlternatingAttention(hidden_dim=hid_d)
        self.p_attention = AlternatingAttention(hidden_dim=hid_d)
        self.s_attention = AlternatingAttention(hidden_dim=hid_d)

        self.w_linear = nn.Linear(hid_d, hid_d)
        self.p_linear = nn.Linear(2*hid_d, hid_d)
        self.s_linear = nn.Linear(2*hid_d, hid_d)

        self.h_linear = nn.Linear(hid_d, ans_list_len)
        ############ 

    def forward(self, image, question_encoding):
        ############ 3.3 TODO
        w_embed, p_embed, s_embed = self.hierQEnc(question_encoding)
        # print(w_embed.shape, p_embed.shape, s_embed.shape)

        v_hat_w, q_hat_w = self.w_attention(w_embed, image)
        v_hat_p, q_hat_p = self.p_attention(p_embed, image)
        v_hat_s, q_hat_s = self.s_attention(s_embed, image)

        temp_w = torch.tanh(self.w_linear(v_hat_w + q_hat_w))
        temp_p = torch.tanh(self.p_linear(torch.cat([v_hat_p + q_hat_p, temp_w], dim=-1)))
        temp_q = torch.tanh(self.s_linear(torch.cat([v_hat_s + q_hat_s, temp_p], dim=-1)))

        return self.h_linear(temp_q)
        ############ 
        # raise NotImplementedError()
