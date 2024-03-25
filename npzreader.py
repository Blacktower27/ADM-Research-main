import numpy as np
import matplotlib.pyplot as plt

i = 10
j = 'p'
data1 = np.load("15mppoVNS.npz")
data2 = np.load("15mVNS.npz")

for array_name in data1:
    array1 = data1[array_name]
# print(array1.shape)
for array_name in data2:
    array2 = data2[array_name]

x = np.linspace(0, 11, 11)  # Adjust the range of x-axis here

# Create two separate plots
plt.figure(figsize=(10, 12))  # Set figure size for better visibility

plt.subplot(2, 1, 1)  # First subplot
plt.plot(x, array1, label='10', color='blue')  # Plot line 1 with label
plt.xlabel('TRAJLEN')  # Set x-axis label
plt.ylabel('objective')  # Set y-axis label
plt.title('PPO_VNS')  # Add title
plt.legend()  # Add legend to identify lines

plt.subplot(2, 1, 2)  # Second subplot
plt.plot(x, array2, label='10', color='orange')  # Plot line 2 with label
plt.xlabel('TRAJLEN')  # Set x-axis label
plt.ylabel('objective')  # Set y-axis label
plt.title('VNS')  # Add title
plt.legend()  # Add legend to identify lines

# Display the plots
plt.tight_layout()  # Adjust spacing between subplots
plt.show()
