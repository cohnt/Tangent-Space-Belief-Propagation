mkdir svg;
mkdir png;
mkdir gif;

for file in *.svg; do inkscape -z -e ${file%svg}png -w 1920 $file; done

mv *.png png
mv *.svg svg

cd ./png

convert dataset.png nearest_neighbors.png pca_observations.png ts_bel_iter*.png -set delay '%[fx:t==(n-1) || t==0 || t==1 || t==2 ? 400 : 20]' belief.gif
convert dataset.png nearest_neighbors.png pca_observations.png ts_mle_iter*.png -set delay '%[fx:t==(n-1) || t==0 || t==1 || t==2 ? 400 : 20]' mle.gif

mv *.gif ../gif
cd ..