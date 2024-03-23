library(magick)

# Produce red, green, and blue matrix data for white, black, most favorite, and least favorite colors
white_red <- matrix(data=255/255, nrow=90, ncol=90)
white_green <- matrix(data=255/255, nrow=90, ncol=90)
white_blue <- matrix(data=255/255, nrow=90, ncol=90)
white_arr <- array(data=c(white_red, white_blue, white_green), dim = c(90,90,3))

black_red <- matrix(data=0, nrow=90, ncol=90)
black_green <- matrix(data=0, nrow=90, ncol=90)
black_blue <- matrix(data=0, nrow=90, ncol=90)
black_arr <- array(data=c(black_red, black_blue, black_green), dim = c(90,90,3))

col1_red <- matrix(data=242/255, nrow=90, ncol=90)
col1_green <- matrix(data=133/255, nrow=90, ncol=90)
col1_blue <- matrix(data=0, nrow=90, ncol=90)
col1_arr <- array(data=c(col1_red, col1_green, col1_blue), dim= c(90,90,3))

col2_red <- matrix(data=1, nrow=90, ncol=90)
col2_green <- matrix(data=1, nrow=90, ncol=90)
col2_blue <- matrix(data=0.4, nrow=90, ncol=90)
col2_arr <- array(data=c(col2_red, col2_green, col2_blue), dim= c(90,90,3))

# Compile the color data into the final image
image <- array(dim = c(180, 180, 3))
image[1:90, 1:90, 1:3] <- white_arr
image[1:90, 91:180, 1:3] <- black_arr
image[91:180, 1:90, 1:3] <- col1_arr
image[91:180, 91:180, 1:3] <- col2_arr

# Plot the image to confirm that the colors are in the right places
plot(magick::image_read(image))

# Show results of filtering the images with each kernel
par(mfrow=c(2, 3))
filt1 <- t(matrix(c(-1,1,0,0), nrow=2, ncol=2))
filt1_image <- array(dim = c(180, 180, 3))
filt1_image <- image_convolve(magick::image_read(image), filt1)
plot(filt1_image)
title(main = "a")

filt2 <- t(filt1)
filt2_image <- array(dim = c(180, 180, 3))
filt2_image <- image_convolve(magick::image_read(image), filt2)
plot(filt2_image)
title(main = "b")

filt3 <- matrix(c(1,1,1,0,0,0,-1,-1,-1), byrow=T, nrow=3, ncol=3)
filt4 <- t(filt3)

filt3_image <- image_convolve(magick::image_read(image), filt3)
plot(filt3_image)
title(main = "c")

filt4_image <- image_convolve(magick::image_read(image), filt4)
plot(filt4_image)
title(main = "d")

filt5 <- matrix(c(0,0,0,1,1,1,0,0,0), nrow=3, ncol=3)
filt6 <- t(filt5)

filt5_image <- image_convolve(magick::image_read(image), filt5)
plot(filt5_image)
title(main = "e")

filt6_image <- image_convolve(magick::image_read(image), filt6)
plot(filt6_image)
title(main = "f")
