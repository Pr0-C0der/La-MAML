echo "Downloading Data..."
# wget -nc http://cs231n.stanford.edu/tiny-imagenet-200.zip (Skipped since in my system wget doesnt work so I manually download and paste it)
echo "Unzipping Data..."
# unzip tiny-imagenet-200.zip

# Downloading and extracting manually using winrar

echo "Last few steps..."
rm -r ./tiny-imagenet-200/test/* #Removes test images and makes the test folder empty
python3 val_data_format.py #Puts validation images into their respective folders
find . -name "*.txt" -delete

