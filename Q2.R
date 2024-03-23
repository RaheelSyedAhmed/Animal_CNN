library(keras)

# Load in CIFAR data
cifar100 <- dataset_cifar100()

# Portion CIFAR data into train and test sets
x_train <- cifar100$train$x
g_train <- cifar100$train$y
x_test <- cifar100$test$x
g_test <- cifar100$test$y

# Scale the training and test features
x_train <- x_train / 255
x_test <- x_test / 255

# Convert labels to categorical values (100 classes present this time)
y_train <- to_categorical(g_train, 100)
y_test <- to_categorical(g_test, 100)

# Build a convolutional model
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3),
                padding = "same", activation = "relu",
                input_shape = c(32, 32, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3),
                padding = "same", activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3),
                padding = "same", activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 256, kernel_size = c(3, 3),
                padding = "same", activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dense(units = 100, activation = "softmax")
summary(model)

# Compile the model
model %>% compile(loss = "categorical_crossentropy",
                  optimizer = optimizer_rmsprop(), metrics = c("accuracy"))

# Fit the model
history <- model %>% fit(x_train, y_train, epochs = 30,
                         batch_size = 128, validation_split = 0.2)
#Print out training and test accuracy
history
# Print out the test accuracy
(model %>% evaluate(x_test, y_test, verbose = F))["accuracy"]


# Produce a better model with dropout and print out the metrics of evaluation
better_model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3),
                padding = "same", activation = "relu",
                input_shape = c(32, 32, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3),
                padding = "same", activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3),
                padding = "same", activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 256, kernel_size = c(3, 3),
                padding = "same", activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(rate = 0.25) %>%
  layer_flatten() %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dense(units = 100, activation = "softmax")
summary(better_model)

better_model %>% compile(loss = "categorical_crossentropy",
                  optimizer = optimizer_rmsprop(), metrics = c("accuracy"))
history2 <- better_model %>% fit(x_train, y_train, epochs = 30,
                         batch_size = 128, validation_split = 0.2)
history2
(better_model %>% evaluate(x_test, y_test, verbose = F))["accuracy"]
