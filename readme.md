# Process to Update H5
1. Delete  existing `train_test.h5` and `data.h5` files.
2. Run `preprocess_and_extract.py` to update train and test data.
3. Run `h5_generator.py` to get the updated train and test data.
4. Run `classifier.py` and see what the new accuracy is.

