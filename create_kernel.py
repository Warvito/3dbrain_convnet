import argparse
import glob
import imp

import numpy as np
import nibabel as nib


def create_npy(args):
    config_name = args.config_name

    try:
        config_module = imp.load_source('config', "./config/" + config_name + ".py")
    except IOError:
        print(
            "Cannot open {}. Please specify the correct name of the configuration file (at the directory ./config). Make sure that the name of the file doesn't have any invalid character and the filename is without the suffix .py at the command. Correct example: python train.py config_test".format(
                config_name))

    paths = config_module.path_files

    labels_file = paths["labels_file"]
    images_dir = paths["raw_images_dir"]
    save_file = paths["python_files"]

    print "Reading labels from %s" % labels_file
    labels = np.genfromtxt(labels_file, delimiter=',', dtype='int8')
    print "   # of labels samples: %d " % len(labels)

    input_data_type = config_module.input_data_type
    print "Reading images with format {} from: %s".format(input_data_type, images_dir)
    paths_train = glob.glob(images_dir + "/*" + input_data_type)
    paths_train.sort()

    n_samples = len(labels)
    if n_samples != len(paths_train):
        raise ValueError('Different number of labels and images files')

    print "Loading images"
    print "   # of images samples: %d " % len(paths_train)
    print ""
    print "{:<5}  {:50s} {:15s}\tCLASS\tMIN    - MAX VALUES".format('#', 'FILENAME', 'DIMENSIONS')
    
    # for testing, reduce to 25 paths
    #paths_train = paths_train[:25]
    n_samples = len(paths_train)

    print n_samples
    
    # allocate memory for a kernel, n subjects x n subjects
    K = np.float64(np.zeros((n_samples, n_samples)))
    
    # number of subjects to read in one block
    step_size = 70
    
    images = []
    
    # outer loop
    for i in range(int(np.ceil(n_samples/np.float(step_size)))) :

	it = i + 1
	max_it = int(np.ceil(n_samples/np.float(step_size)))

	print " outer loop iteration: %d of %d." % (it, max_it)
    
        # generate indices and then paths for this block
        start_ind_1 = i * step_size
        stop_ind_1 = min(start_ind_1 + step_size, n_samples)       
        block_paths_1 = paths_train[start_ind_1:stop_ind_1]
        
        # read in the images in this block
        images_1 = []
        for k, path in enumerate(block_paths_1):
            img = nib.load(path)
            img = img.get_data()
            img = np.asarray(img, dtype='float64')
	    #print np.shape(img)
            #print "{:<5}  {:50s} ({:3}, {:3}, {:3})\t{:}\t{:6.4} - {:6.4}".format((k + 1), os.path.basename(os.path.normpath(path)),
                                                                   #img.shape[0], img.shape[1], img.shape[2], labels[k],
                                                                   #np.min(img), np.max(img))
 
                                                                   
            # THIS IS WHERE MASKING, WITHIN SUBJECT IMAGE SELECTION ETC MUST BE DONE
                                                                   
            # reshape image data to a vector and add to images
            img_vec = np.reshape(img, np.product(img.shape)) 
            #print np.max(img_vec)
	    #print np.min(img_vec)
	    #print np.sum(img_vec)
	    #print len(img_vec)                                            
            images_1.append(img_vec)
            del img
	images_1 = np.array(images_1)
	
            
        # inner loop
        #for j in range(len(n_samples/step_size) + 1) :
        for j in range(i + 1) :

	    it = j + 1
	    max_it = i + 1	

	    print " inner loop iteration: %d of %d." % (it, max_it)
            
            # if i = j, then sets of image data are the same - no need to load
            if i == j :
                
                start_ind_2 = start_ind_1
                stop_ind_2 = stop_ind_1
                images_2 = images_1
                
            # if i !=j, read in a different block of images
            else :
                
                # generate indices and then paths for this block
                start_ind_2 = j * step_size
                stop_ind_2 = min(start_ind_2 + step_size, n_samples)       
                block_paths_2 = paths_train[start_ind_2:stop_ind_2]
        
                # read in the images in this block
                images_2 = []
                for k, path in enumerate(block_paths_2):
                    img = nib.load(path)
                    img = img.get_data()
                    img = np.asarray(img, dtype='float64')
                    #print "{:<5}  {:50s} ({:3}, {:3}, {:3})\t{:}\t{:6.4} - {:6.4}".format((k + 1), os.path.basename(os.path.normpath(path)),
                     #                                              img.shape[0], img.shape[1], img.shape[2], labels[k],
                      #                                             np.min(img), np.max(img))
                                                                   
                    # THIS IS WHERE MASKING, WITHIN SUBJECT IMAGE SELECTION ETC MUST BE DONE
                                                                   
                    # reshape image data to a vector and add to images
                    img_vec = np.reshape(img, np.product(img.shape))                                                      
                    images_2.append(img_vec)
                    del img
		images_2 = np.array(images_2)
                    
            # fill in the kernel matrix with a dot product of the image blocks
            block_K = np.dot(images_1, np.transpose(images_2))
	    #print block_K[0, 0]
	    #print sum(images_1[0,:] * images_2[0,:])
            K[start_ind_1:stop_ind_1, start_ind_2:stop_ind_2] = block_K
            K[start_ind_2:stop_ind_2, start_ind_1:stop_ind_1] = np.transpose(block_K)
            
    print ""
    print "Saving Dataset"
    print "   Kernel+Labels:" + save_file
    np.savez(save_file, kernel=K, labels=labels)
    del images
    print "Done"
    print save_file

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to create dataset files.')
    parser.add_argument("config_name", type=str, help="The name of file .py with configurations, e.g., Alexnet")
    args = parser.parse_args()
    create_npy(args)

