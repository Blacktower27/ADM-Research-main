import numpy as np
import matplotlib.pyplot as plt

i = 5
j = 'p'
data1 = np.load("Results/ACF%d-SC%c" % (i, j) + '/NTFres_PPO.npz')
data2 = np.load("Results/ACF%d-SC%c" % (i, j) + '/res_PPO.npz')

for array_name in data1:
    array1 = data1[array_name]

for array_name in data2:
    array2 = data2[array_name]

x = np.linspace(0, 5000, 5000)  # Adjust the range of x-axis here

# Create two separate plots
plt.figure(figsize=(6, 12))  # Set figure size for better visibility

plt.subplot(2, 1, 1)  # First subplot
plt.plot(x, array1, label='Normal', color='blue')  # Plot line 1 with label
plt.xlabel('eposode')  # Set x-axis label
plt.ylabel('objective')  # Set y-axis label
plt.title('Normal')  # Add title
plt.legend()  # Add legend to identify lines

plt.subplot(2, 1, 2)  # Second subplot
plt.plot(x, array2, label='Transfer learning', color='orange')  # Plot line 2 with label
plt.xlabel('eposode')  # Set x-axis label
plt.ylabel('objective')  # Set y-axis label
plt.title('Transfer Learning')  # Add title
plt.legend()  # Add legend to identify lines

# Display the plots
plt.tight_layout()  # Adjust spacing between subplots
plt.show()
