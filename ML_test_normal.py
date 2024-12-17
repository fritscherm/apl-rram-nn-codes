#import datafile_functions as df
#import datafile_processing_functions as dpf
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model
import tensorflow.keras.metrics
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
import sys
import pickle

if len(sys.argv) < 2:
    print("no length specified")
    curfrac = 1
else:
    curfracint = int(sys.argv[1])
    curfrac = curfracint / 1000
    #curfrac = curfracint / 100

#Load list of all the files
#file_list = df.Datafile_Load('/scratch/disk0/user/project/newdata/MLP', filter_out=['sweep', 'var', '.png', 'desktop']).load()[0]

def load_par(file_list):
    """
    Load the pulse parameters of each file
    Remove any files from the list of files that cannot be loaded due to error
    
    Parameters
    ----------
    file_list : List containing all the data files to be processed
    
    Returns
    -------
    List containing parameters of each file
    New list of files with valid parameters
    """
    
    par = list()
    err = list()
    
    for f in tqdm(file_list):
        try:
            par.append(dpf.parameters_auto(f))
        
        except:
            err.append(f)
    
    new_list = [x for x in file_list if x not in err]
    
    return new_list, par
  
def extract_par_val(all_files, parameters):
    """
    Extract values from parameters list and store them into separate lists
    
    Parameters
    ----------
    all_files : List containing all the data files to be processed
    parameters : List containing the parameter values to be stored
    
    Returns
    -------
    Individual lists of the seven parameter values for each file
    """
    
    current = list()
    pulse_number = list()
    writing_pulse = list()
    writing_w = list()
    d_width = list()
    r_width = list()
    anti_pulse = list()

    for f, p in tqdm(zip(all_files, parameters)):
        if 'pulse_num' in p[0][0].keys():
            y_idx_plot = dpf.Pulse_Plot_Info(f).idx_to_plot()[0]
            io = dpf.Load_Data(f).load()[2][y_idx_plot].reshape(-1)
            current.extend(io)
        
            to_all = dpf.Load_Data(f).load()[4]
            trig_per = np.round(to_all[1]-to_all[0], decimals=3)
            for c in range(y_idx_plot.shape[0]):
                pulse_number.append(c)
                writing_pulse.append(p[1][0]['v_set'])
                d_width.append(p[1][1]['d_width']*trig_per)
                r_width.append(p[1][1]['r_width']*trig_per)
                if 'v_set_fb' in p[1][0].keys(): 
                    anti_pulse.append(p[1][0]['v_set_fb'])
                    writing_w.append(p[1][1]['v_set_width(p+fb)'][0]*trig_per)
                else: 
                    anti_pulse.append(0)
                    writing_w.append(p[1][1]['v_set_width']*trig_per)
    
    if 'pulse_1_num' in p[0][0].keys():
        y_idx_plot = dpf.Pulse_Plot_Info(f).idx_to_plot()[0]
        sec_idx = p[1][2]['r_start_2_idx']
        set_end_idx = np.argwhere(y_idx_plot==sec_idx)[0][0]
        io_set = dpf.Load_Data(f).load()[2][y_idx_plot[:set_end_idx]].reshape(-1)
        io_rst = dpf.Load_Data(f).load()[2][y_idx_plot[set_end_idx:]].reshape(-1)
        
        current.extend(io_set)
        current.extend(io_rst)
        
        to_all = dpf.Load_Data(f).load()[4]
        trig_per = np.round(to_all[1]-to_all[0], decimals=3)[0]
        for c in range(io_set.shape[0]):
            pulse_number.append(c)
            writing_pulse.append(p[1][0]['v_set'])
            d_width.append(p[1][1]['d_width']*trig_per)
            r_width.append(p[1][1]['r_width']*trig_per)
            if 'v_set_fb' in p[1][0].keys(): 
                anti_pulse.append(p[1][0]['v_set_fb'])
                writing_w.append(p[1][1]['v_set_width(p+fb)'][0]*trig_per)
            else: 
                anti_pulse.append(0)
                writing_w.append(p[1][1]['v_set_width']*trig_per)

        for c in range(io_rst.shape[0]):
            pulse_number.append(c)
            writing_pulse.append(p[1][0]['v_rst'])
            d_width.append(p[1][1]['d_width']*trig_per)
            r_width.append(p[1][1]['r_width']*trig_per)
            if 'v_rst_fb' in p[1][0].keys(): 
                anti_pulse.append(p[1][0]['v_rst_fb'])
                writing_w.append(p[1][1]['v_rst_width(p+fb)'][0]*trig_per)
            else: 
                anti_pulse.append(0)
                writing_w.append(p[1][1]['v_rst_width']*trig_per)
    
    return current, pulse_number, writing_pulse, writing_w, d_width, r_width, anti_pulse

def create_dataframe(current, pulse_number, writing_pulse, writing_w, d_width, r_width, anti_pulse):
    dataframe = pd.DataFrame()
    
    dataframe['y'] = current
    dataframe['Pulse Number'] = pulse_number
    dataframe['Writing Pulse'] = writing_pulse
    dataframe['Anti-pulse'] = anti_pulse
    dataframe['Writing Pulse Width'] = list(map(float, writing_w))
    dataframe['Delay Width'] = list(map(float, d_width))
    dataframe['Reading Width'] = list(map(float, r_width))

    return dataframe

def norm(x, y):
    feature_mean = x.mean(axis=0)
    feature_std = x.std(axis=0)
    
    x_norm = (x-feature_mean)/feature_std
    y_norm = np.empty(y.shape[0])
    for i in range(int(y.shape[0]/100)):
        y_norm[i*100:100+i*100] = minmax_scale(y[i*100:100+i*100])
        
    return x_norm, y_norm, feature_mean, feature_std

# we omit the following in order to avoid cluttered code since we use a lot of libraries. Instead, we provide and load the resulting pickle file.

#all_files, parameters = load_par(file_list)
#I, pulse_num, V_wr, v_width, d_width, r_width, V_anti = extract_par_val(all_files, parameters)
#df = create_dataframe(I, pulse_num, V_wr, v_width, d_width, r_width, V_anti)
#
#values = df.values
#x = values[:,1:] #Store the feature values except for the pulse number
#y = values[:,0] #Current values

xfile = open("x.pck", rb)
x = pickle.load(xfile)
xfile.close
yfile = open("y.pck", rb)
y = pickle.load(yfile)
yfile.close

X, Y, feature_mean, feature_std = norm(x, y)

def build_MLP(shape=X.shape[1], neurons=128, hidden_layers=1, droprate=.5, learning_rate=1e-3):

  input_ = layers.Input(shape=(shape))
  x = layers.Dense(neurons, activation='relu')(input_)
  x = layers.Dropout(droprate)(x)
  
  for _ in range(hidden_layers-1):
    x = layers.Dense(neurons, activation='relu')(x)
    x = layers.Dropout(droprate)(x)

  output_ = layers.Dense(1, activation='linear')(x)

  mlp = Model(input_, output_)
  print(mlp.summary())

  mlp.compile(
      loss='mse',
      optimizer=Adam(learning_rate=learning_rate),
      metrics=[tensorflow.keras.metrics.MeanSquaredError()]
  )

  return mlp

def plot_history(history):
  fig, ax = plt.subplots()

  # create error sublpot
  ax.plot(history.history["loss"], label="train error")
  ax.plot(history.history["val_loss"], label="test error")
  ax.set_ylabel("Error")
  ax.set_xlabel("Epoch")
  ax.legend(loc="upper right")
  # ax.set_title("Error eval")

#Split into training and test data sets  
x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.25, random_state=42)

print("xtrainshape:")
print(x_train.shape[0])
datalen = x_train.shape[0]
curlen = int(curfrac*datalen)
print(x_train.shape)
print(y_train.shape)
x_train = x_train[:curlen,:]
y_train = y_train[:curlen]
print(x_train.shape)
print(y_train.shape)
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(buffer_size=100).batch(64)
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(64)

mlp = build_MLP(shape=(6,), neurons=64, hidden_layers=1, droprate=.2, learning_rate=1e-4)

history = mlp.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=1000,
    verbose=0,
    )
saveheader ="mnormalsmallstep"
export_path = "/scratch/disk0/user/project/nets/" + saveheader + "-" + str(curfrac) + "-network.pickle"

with open(export_path, "wb") as pickle_file:
    pickle.dump(mlp, pickle_file)

#plot_history(history)
for curdata in ["val_loss", "loss", "mean_squared_error", "val_mean_squared_error"]:
    a = history.history[curdata]
    print(a)
    savestring = saveheader + str(curfrac) + "-data-" +  str(curdata) + ".csv"
    np.savetxt(savestring, a, delimiter=',')
'--------------------------------------------------------------------------------'
#TESTING
for testsa in range (1,10):
    test_idx = testsa    
    store = np.empty(100)
    for i in range(100):
        test_sample = np.expand_dims( (np.insert(x[test_idx,1:], 0, i) - feature_mean) / feature_std, 0)
    
        store[i] = mlp.predict(test_sample)
    
    fig = plt.figure()
    plt.plot(store, '--o', marker = '.', markersize=1, linestyle='none', label='Predicted')
    plt.plot(Y[test_idx*100:100+test_idx*100],'--o', marker = '.', markersize=1, linestyle='none', label='Empirical')
    plt.xlabel('Pulse Number')
    plt.ylabel('Scaled Conductivity [S]')
    plt.legend(loc='lower right', fontsize=8, markerscale=6)
    savestring = "/scratch/disk0/user/project/outimg/" + str(testsa) +"/" + saveheader + "-" + str(curfrac) + '.png'
    plt.savefig(savestring)
    plt.close(fig)
    #also storing data as csv
    savestring = "/scratch/disk0/user/project/outimg/" + str(testsa) +"/" + saveheader + "-" + str(curfrac) + '-emp.csv'
    np.savetxt(savestring, Y[test_idx*100:100+test_idx*100], delimiter=',')
    savestring = "/scratch/disk0/user/project/outimg/" +  str(testsa) +"/" + saveheader + "-" + str(curfrac) + '-pred.csv'
    np.savetxt(savestring, store, delimiter=',')

