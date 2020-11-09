for filename in ../videoSamples/positiveSamples/*.mp4; do
    ffmpeg -i $filename -vcodec libx264 $filename -y
done

for filename in ../videoSamples/negativeSamples/*.mp4; do
    ffmpeg -i $filename -vcodec libx264 $filename -y
done