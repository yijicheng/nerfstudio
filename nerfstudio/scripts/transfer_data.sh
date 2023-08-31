mkdir compressed_folder  
while read line; do tar -czvf "compressed_folder/$line.tar.gz" "$line"; done < 文件名.txt