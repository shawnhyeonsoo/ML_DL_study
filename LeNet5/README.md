Paper: Gradient-based Learning Applied to Document Recognition </br>
> Yann Lecun et. al </br>
 </br>

Total 7 Layers: </br>
- Layer C1: Convolutional Layer with 6 filters of size 5 x 5, producing 6 feature maps of size 28 x 28 
- Layer S2: Average Pooling, 6 filters of size 2 x 2, producing 6 feature maps of size 14 x 14
- Layer C3: Convolutional Layer with 16 filters of size 5 x 5, producing 16 feature maps of size 10 x 10
- Layer S4: Average Pooling, 16 filters of size 2 x 2, producing 16 feature maps of size 5 x 5
- Layer C5: Convolutional Layer with 120 filters of size 5 x 5, producing 120 feature maps of size 1 x 1
- Layer F6: Fully- Connected Layer to produce 84 feature maps of size 1 x 1
- Layer Output: Layer to classify into 10 classes

</br>

LENET5 </br>
Accuracy: 0.9894
