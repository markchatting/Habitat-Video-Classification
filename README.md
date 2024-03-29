# Habitat-Video-Classification
Classify underwater videos into habitat classes

These scripts automate the classification of hours worth of benthic video towed data. This repository is made up of several scripts to perform various tasks of the whole process:
  1)  Images were extracted from each video
  2)  Images were sorted into categories then the habitat model was trained on these images
  3)  Coral reef, seagrass and algae images were also categorized into dense vs sparse to train another model to differentiate between dense and sparse
  4)  Habitat classifications for each second in the videos was predicted along with other outputs requested by the client:
    i)  For each second of video the client needed data displayed on the videoscreen. This included Coordinates, water temperature and depth
    ii) For each second of video hatbitat was classified along with coordinates and other data being read from the display by pytesseract.
  
These scripts clasify underwater videos based on two CNN models. The first model classifies habitat only (seagrass, macro algae, coral reef, offshore sand sheets etc.) and if seagrass, coral or algae is classified, the second model classifies density (dense vs sparse). Overall the habitat model has a ~85% classifying accuracy and the density model is ~88% when the videos are clear. There is also a ‘catch all’ category for the habitat model (‘other’) for when videos weren’t very clear, which helped increase the clear video classification accuracy. The videos also contain various data reported onscreen that needed to be captured. Pytesseract reads the text data onscreen and regular expressions were used to parse out depth, coordinates (lon and lat) and water temperature from the videos.

Also the rate that it works is really slow. I can speed it up but it’ll sacrifice accuracy a bit. To explain:
a.	The way this program works is it slices up each video into frames and then classifies each frame. The classifying speed is about 1 second per frame (this can’t really be changed) and I think the videos are ~30 frames per second, so a 10 minute video will have around 18,000 images and take around 5 hours to classify. But this 5 hours of classifying increases the accuracy to >90%. The way I did that was after all images had been classified I grouped them by unique lat and lon pairings and then the habitat that was classified the most for each coordinate pair was chosen. Basically, it’s a voting system for each unique lat and lon. Because I was classifying every single frame there were a lot of votes for the same lat and lon and any mistakes were overridden by the most frequent classification.
b.	I can skip forward a certain number of frames (I’ll call that image rate) to speed up the whole process, but that’ll give fewer votes for each lat and lon pair so more chances for mistakes to not get overridden. If I skip forward every 30 frames (~1 second in video time) it’ll roughly take the same speed to classify as it does to just watch each video but the accuracy will drop down to the original model accuracy (~85%).

The user interface looks a little 1990’s and there’s a couple issues with it. First, after a video has finished being classified, the “Choose Video” buttons don’t work. They only work the first time you press them. You need to close the program and start it up again for them to work. It’s really been annoying me and I’ve been trying for ages to fix it but I haven’t managed to yet. If anyone has any suggestions on how to fix that i'd love to hear them.
