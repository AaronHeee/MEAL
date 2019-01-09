URL=https://www.dropbox.com/s/ruh0esxg83z3i3u/checkpoint.zip
ZIP_FILE=./checkpoint.zip
wget -N $URL -O $ZIP_FILE
unzip $ZIP_FILE -d ./
rm $ZIP_FILE