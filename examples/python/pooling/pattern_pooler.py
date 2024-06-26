# ==============================================================================
# anom_pooler.py
# ==============================================================================
from brainblocks.blocks import ScalarTransformer, PatternPooler, SequenceLearner
import numpy as np

# Printing boolean arrays neatly
np.set_printoptions(
    precision=3, suppress=True, threshold=1000000, linewidth=80,
    formatter={"bool": lambda bin_val: "X" if bin_val else "-"})

train_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
test_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Setup blocks
st = ScalarTransformer(
    min_val=0.0, # minimum input value
    max_val=1.0, # maximum input value
    num_s=128,  # number of statelets
    num_as=16)  # number of active statelets

pp = PatternPooler(
    num_s=128,     # number of statelets
    num_as=16,      # number of active statelets
    perm_thr=20,   # receptor permanence threshold
    perm_inc=2,    # receptor permanence increment
    perm_dec=1,    # receptor permanence decrement
    pct_pool=0.8,  # pooling percentage
    pct_conn=0.5,  # initially connected percentage
    pct_learn=0.3) # learn percentage

# Connect blocks
pp.input.add_child(st.output, 0)

# train through the values 3 times
for iteration_i in range(3):
    for i in range(len(train_values)):

        # Set scalar transformer value
        st.set_value(train_values[i])

        # Compute the scalar transformer
        st.feedforward()

        # Compute the pattern pooler
        pp.feedforward(learn=True)

prev_scalar = np.array(None)
prev_pool = np.array(None)

for i in range(len(test_values)):

    # Set scalar transformer value
    st.set_value(test_values[i])

    # Compute the scalar transformer
    st.feedforward()

    # Compute the pattern pooler
    pp.feedforward(learn=False)

    # List of 0s and 1s representing the distributed binary representation
    intarray = st.output.bits

    # Converted to numpy array for visualization
    scalar_array = np.array(st.output.bits, dtype=bool)
    pool_array = np.array(pp.output.bits, dtype=bool)

    # compare current value and encodings with previous
    if i > 0:
        float_diff = test_values[i]-test_values[i-1]
        scalar_sim = np.dot(scalar_array.astype(int), prev_scalar.astype(int))
        pool_sim = np.dot(pool_array.astype(int), prev_pool.astype(int))
        result = f"{test_values[i]} {float_diff:.2f} {scalar_sim} {pool_sim}"
        print(result)
    else:
        print(f"{test_values[i]}")
    # print(scalar_array)
    # print(pool_array)

    # save the scalar and pooling codes to compare similarity with adjacent encodings
    prev_scalar = scalar_array
    prev_pool = pool_array

