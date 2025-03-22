import numpy as np

def generate_m1_m2(max_value, step):
    # Define the range for m1 and m2
    num = max_value/step
    m1_values = step * np.arange(0, num + 1, 1)
    # Prepare lists to store m1 and m2 values
    m1_result = []
    m2_result = []

    # Generate the nested parameter values for m1 and m2
    for m1 in m1_values:
        m2_values = step * np.arange(m1/step, num + 1, 1)
        m1_result.extend([m1] * len(m2_values))
        m2_result.extend(m2_values)

    # Format the output as required text format
    m1_text = ", ".join([f"{m1:.5f}" for m1 in m1_result])
    m2_text = ", ".join([f"{m2:.5f}" for m2 in m2_result])

    # Create the final text output
    final_output = f"m1 {m1_text} []\nm2 {m2_text} []"
    return final_output


# Example usage
# max_value = 0.12
# step = 0.005
# max_value = 0.06
# step = 0.001
max_value = 0.012
step = 0.0002
result = generate_m1_m2(max_value, step)
print(result)
