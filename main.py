import libemg
import os
import zipfile
import scipy.io
import numpy as np


# given a directory, return a list of files in that directory matching a format
# can be nested
def find_all_files_of_type_recursively(dir, terminator):
    files = os.listdir(dir)
    file_list = []
    for file in files:
        if file.endswith(terminator):
            file_list.append(dir+file)
        else:
            if os.path.isdir(dir+file):
                file_list += find_all_files_of_type_recursively(dir+file+'/',terminator)
    return file_list

# convert the file to .csv format -- also downsample from 2kHz to 1kHz
def convert_to_csv(mat_file):
    # read the mat file
    mat_dir = mat_file.split('/')
    mat_dir = mat_dir[0] +'/' + mat_dir[1] + '/'
    mat = scipy.io.loadmat(mat_file)
    # get the data
    subject = mat['subject'][0][0]
    exercise = int(mat_file.split('_')[3][1])
    if exercise == 1:
        exercise_offset = 0 # 0 reps already included
    elif exercise == 2:
        exercise_offset = 10 # 10 reps already included
    elif exercise == 3:
        exercise_offset = 20 # 18 reps already included
    data = mat['emg']
    restimulus = mat['restimulus']
    rerepetition = mat['rerepetition']
    # remove 0 repetition - collection buffer
    remove_mask = (rerepetition != 0).squeeze()
    data = data[remove_mask,:]
    restimulus = restimulus[remove_mask]
    rerepetition = rerepetition[remove_mask]
    # important little not here: 
    # the "rest" really is only the rest between motions, not a dedicated rest class.
    # there will be many more rest repetitions (as it is between every class)
    # so usually we really care about classifying rest as its important (most of the time we do nothing)
    # but for this dataset it doesn't make sense to include (and not its just an offline showcase of the library)
    # I encourage you to plot the restimulus to see what I mean. -> plt.plot(restimulus)
    # so we remove the rest class too
    remove_mask = (restimulus != 0).squeeze()
    data = data[remove_mask,:]
    restimulus = restimulus[remove_mask]
    rerepetition = rerepetition[remove_mask]
    tail = 0
    while tail < data.shape[0]-1:
        rep = rerepetition[tail][0] # remove the 1 offset (0 was the collection buffer)
        motion = restimulus[tail][0] # remove the 1 offset (0 was between motions "rest")
        # find head
        head = np.where(rerepetition[tail:] != rep)[0]
        if head.shape == (0,): # last segment of data
            head = data.shape[0] -1
        else:
            head = head[0] + tail
        # downsample to 1kHz from 2kHz using decimation
        data_for_file = data[tail:head,:]
        data_for_file = data_for_file[::2, :]
        # write to csv
        csv_file = mat_dir + 'C' + str(motion-1) + 'R' + str(rep-1 + exercise_offset) + '.csv'
        np.savetxt(csv_file, data_for_file, delimiter=',')
        tail = head

    os.remove(mat_file)


# The dataset should be downloaded and placed in the dataset folder.
# the subject files should each be a zipped file at this stage
def prepare_data(dataset_dir):
    # get the zip files (original format they're downloaded in)
    zip_files = find_all_files_of_type_recursively(dataset_dir,".zip")
    # unzip the files -- if any are there (successive runs skip this)
    for zip_file in zip_files:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(zip_file[:-4]+'/')
        os.remove(zip_file)
    
    
    # get the mat files (the files we want to convert to csv)
    mat_files = find_all_files_of_type_recursively(dataset_dir,".mat")
    for mat_file in mat_files:
        file_parts = mat_file.split('/')[2].split('_')
        subject_num = int(file_parts[0][1:])
        trial_num = int(file_parts[2][1])
        convert_to_csv(mat_file)
    

    





if __name__ == "__main__":
    dataset_dir = 'dataset/'
    # prepare means get it into a format that libemg can consume it -- 
    # a collection of .csv files
    prepare_data(dataset_dir)