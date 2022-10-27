#! /bin/bash

for file in **/*.wav;do
  outfile="${file%.*}.png"
  sox "$file"  -c 1 -t wav - | sox -t wav - -n spectrogram -o "$outfile"
done



