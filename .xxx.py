import statistics

# Get three numbers from user input
num1 = 0.4732
num2 = 0.4766
num3 = 0.4782

# Create a list of the numbers
numbers = [num1, num2, num3]

# Calculate mean and standard deviation
mean = statistics.mean(numbers)
stdev = statistics.stdev(numbers)

# Display results
print(f"{mean:.4f}±{stdev:.4f}")