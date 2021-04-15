# Colorization_deeplearning
<p>The final thesis for me in the AUT around colorizing the grayscale CT images. Two deep learning networks are utilized to generate the colorful CT lung images. </p>
<p>VGG-19 and ResNet based on exemplar colorization and full-automatic colorization, respectively. </p>
<p>As for exemplar colorization, the crucial question is to select appropriate reference images to extract the colours for the target images. Due to the colours of meat from animals is resemble the human lungs, so fresh pork, lamb, beef, and even rotten meat (for infected lung) are collected to prepare for the models transferring the style and texture to the target images. </p>
<p>
The video displays the final results in each iterations：
<a href="https://www.youtube.com/watch?v=0GAjLw6MXMo/">click this link</a>
</p>
<p>For another fully automatic approach, two sets of training data individually comprising painting work and meat pictures are implemented to obtain the per-pixel erudition for the CT lung image colorizations.</p>
