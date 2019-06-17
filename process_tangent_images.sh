mkdir svg;
mkdir png;
mkdir gif;

for file in *.svg; do inkscape -z -e ${file%svg}png -h 1080 $file; done

mv *.png png
mv *.svg svg

cd ./png

convert dataset.png nearest_neighbors.png pca_observations.png ts_bel_iter*.png pruned_nearest_neighbors.png added_edges.png \
-set delay '%[fx:t==(n-1) || t==(n-2) || t==(n-3) || t==0 || t==1 || t==2 ? 400 : 20]' belief.gif

convert dataset.png nearest_neighbors.png pca_observations.png ts_mle_iter*.png pruned_nearest_neighbors.png added_edges.png \
-set delay '%[fx:t==(n-1) || t==(n-2) || t==(n-3) || t==0 || t==1 || t==2 ? 400 : 20]' mle.gif

convert error_histogram*.png -set delay '%[fx:t==(n-1) || t==0 ? 400 : 50]' error_histogram.gif

convert dataset.png nearest_neighbors.png pca_observations.png ts_mle_iter*.png pruned_nearest_neighbors.png added_edges.png coord_bp.png \
-set delay '%[fx:t==(n-1) || t==(n-2) || t==(n-3) || t==(n-3) || t==0 || t==1 || t==2 ? 400 : 20]' result.gif

mv *.gif ../gif
cd ..