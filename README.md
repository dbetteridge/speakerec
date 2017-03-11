# speakerec

Takes a series of wav files of human speech and performs FFT on them, 
the frequency domain data is then passed into a feedforward NN and used to train it.

Minimum 100 epochs to a reasonable convergence for 3+ speakers

outputs array is used to define which speech sample corresponds to a person i.e
Bob,John, Suzy

if the sample is from Bob , output = [1,0,0]
John , output = [0,1,0]
Suzy, output = [0,0,1]
