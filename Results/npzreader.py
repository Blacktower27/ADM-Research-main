import numpy as np
import matplotlib.pyplot as plt

i = 10
j = 'p'
data1 = np.load("/Users/blairlu/code/python_code/intern/ADM-Research-main/Results/ACF15-SCm/res_distance.npz")
data2 = np.load("/Users/blairlu/code/python_code/intern/ADM-Research-main/Results/ACF15-SCm/res_distance10.npz")

for array_name in data1:
    array1 = data1[array_name]

for array_name in data2:
    array2 = data2[array_name]

x = np.linspace(0, 1000, 100)  # Adjust the range of x-axis here

# Create two separate plots
plt.figure(figsize=(10, 12))  # Set figure size for better visibility

plt.subplot(2, 1, 1)  # First subplot
plt.plot(x, array1, label='100', color='blue')  # Plot line 1 with label
plt.xlabel('eposode')  # Set x-axis label
plt.ylabel('objective')  # Set y-axis label
plt.title('100')  # Add title
plt.legend()  # Add legend to identify lines

plt.subplot(2, 1, 2)  # Second subplot
plt.plot(x, array2, label='10', color='orange')  # Plot line 2 with label
plt.xlabel('eposode')  # Set x-axis label
plt.ylabel('objective')  # Set y-axis label
plt.title('10')  # Add title
plt.legend()  # Add legend to identify lines

# Display the plots
plt.tight_layout()  # Adjust spacing between subplots
plt.show()
