from datetime import datetime
import os
import numpy as np


def save_results(PARAMS,train_loss,validation_loss,validation_fscore):

    print("test")
    
    arrays_dict = {
    'train_loss': train_loss,
    'validation_loss': validation_loss,
    'validation_fscore': validation_fscore
    }
    
 
    # Get the current date and time
    now = datetime.now()
    date_time_folder = now.strftime("%d_%m_%H:%M")

    # Create a new directory for today's date and time inside the 'results' folder
    results_directory = './results/{}'.format(date_time_folder)
    os.makedirs(results_directory, exist_ok=True)

    # Save the parameters in the 'results' directory
    parameters_file_path = os.path.join(results_directory, 'parameters.txt')
    with open(parameters_file_path, 'w') as file:
        for key, value in PARAMS.items():
            file.write(f'{key}: {value}\n')

    # Save each array in the new directory
   
        for name, array in arrays_dict.items():
            array_file_path = os.path.join(results_directory, f'{name}.npy')
            np.save(array_file_path, array)
        
        

   
