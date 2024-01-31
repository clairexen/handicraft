# http://stackoverflow.com/a/22535383/2213720

array=( [0]=24 [1]=24 [5]=10 [6]=24 [10]=24 [12]=12 )
ref=( )

for i in "${!array[@]}"; do
    ref[array[i]]="${ref[array[i]]}$i "
done

for i in "${!ref[@]}"; do
    [[ "${ref[i]% }" == *" "* ]] && echo "$i @ ${ref[i]% }"
done
