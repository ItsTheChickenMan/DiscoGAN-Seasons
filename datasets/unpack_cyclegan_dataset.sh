FILE=$1

echo "Specified [$FILE]"

ZIP_FILE=./datasets/$FILE.zip
mkdir $TARGET_DIR
unzip $ZIP_FILE -d ./datasets/
rm $ZIP_FILE