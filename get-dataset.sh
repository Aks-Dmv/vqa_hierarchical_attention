sudo apt install -y pv

export DATA=/mnt/disks/hw3-data/

ANNO_TRN_ZIP=Annotations_Train_mscoco.zip
ANNO_VAL_ZIP=Annotations_Val_mscoco.zip
QUES_TRN_ZIP=Questions_Train_mscoco.zip
QUES_VAL_ZIP=Questions_Val_mscoco.zip
IMGS_TRN_ZIP=train2014.zip
IMGS_VAL_ZIP=val2014.zip

AWS_COCO_LST="$ANNO_TRN_ZIP $ANNO_VAL_ZIP $QUES_TRN_ZIP $QUES_VAL_ZIP"
ORG_COCO_LST="$IMGS_TRN_ZIP $IMGS_VAL_ZIP"

AWS_COCO_DIR=https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/
ORG_COCO_DIR=http://images.cocodataset.org/zips/

ANNO_TRN=mscoco_train2014_annotations.json
ANNO_VAL=mscoco_val2014_annotations.json
QUES_TRN=OpenEnded_mscoco_train2014_questions.json
QUES_VAL=OpenEnded_mscoco_val2014_questions.json
IMGS_TRN=train2014/
IMGS_VAL=val2014/

# check if zips exist in directory, if not, download from internet
get_zips() {
	# if [ $# != 3 ]; then
	# 	echo "get_zips must be given 2 arguments"
	# 	return 1
	# fi

	for zip in $1
	do
		fname="$DATA/$zip"
		if [ ! -f $fname ]; then
			echo "downloading $fname from $2$zip ..."

			wget -P $DATA $2$zip
			if [ -f $fname ]; then
				echo "downloaded."
			else
				echo "something went wrong with $fname, skipping ..."
			fi

		else
			echo "$fname already exists"
		fi
		printf "\n"
	done
	return 0
}

# check if DATA mountpoint exists
if [ -d "$DATA" ]
then
	echo "$DATA directory exists"
	printf "\n"
else
	echo "ERR: $DATA directory does not exist, please mount the data disk to $DATA"
	exit 1
fi

# get annotations and questions for train and val
get_zips "$AWS_COCO_LST" "$AWS_COCO_DIR"
# get images for train and val
get_zips "$ORG_COCO_LST" "$ORG_COCO_DIR"

# extract annotations and questions unconditionally
echo "extracting $AWS_COCO_LST ..."
for zip in $AWS_COCO_LST
do
	unzip -q -o "$DATA/$zip" -d $DATA 
done

# extract training data
if [ ! -d "$IMGS_TRN" ]; then
	echo "extracting $IMGS_TRN_ZIP ..."
	nfiles=$(unzip -l "$DATA/$IMGS_TRN_ZIP" | grep .jpg | wc -l)
	unzip -o "$DATA/$IMGS_TRN_ZIP" -d $DATA | pv -l -s $nfiles > /dev/null
else
	echo "$IMGS_TRN already extracted"
fi

#extract validation data
if [ ! -d "$IMGS_VAL" ]; then
	echo "extracting $IMGS_VAL_ZIP ..."
	nfiles=$(unzip -l "$DATA/$IMGS_VAL_ZIP" | grep .jpg | wc -l)
	unzip -o "$DATA/$IMGS_VAL_ZIP" -d $DATA | pv -l -s $nfiles > /dev/null
else
	echo "$IMGS_VAL already extracted"
fi

