import pathlib
import pickle
from time import time

from brainblocks.blocks import ContextLearner, DiscreteTransformer
import numpy as np
# convert letters to integer labels instead of byte values
from sklearn import preprocessing

# printing boolean arrays neatly
np.set_printoptions(precision=3, suppress=True, threshold=1000000, linewidth=300,
                    formatter={'bool': lambda bin_val: 'X' if bin_val else '-'})


def reduce_letters(text_data):
    """
    Reduce the number of letters to consider, so we don't allocate states to chars we never see
    :param text_data:
    :return:
    """

    # string to a list of char
    text_array = [*text_data]

    # Use scikit-learn to convert letters to discrete integers
    char_to_int_encoder = preprocessing.LabelEncoder()
    char_to_int_encoder.fit(text_array)

    # create the input you would use for your machine learning model
    text_as_integers = char_to_int_encoder.transform(text_array)

    num_letters = len(char_to_int_encoder.classes_)
    print(len(text_as_integers), "total characters")
    print(f"{num_letters} distinct characters")

    # get the count of each character
    # unique, counts = np.unique(text_array, return_counts=True)
    # all_freq = dict(zip(unique, counts))

    # show each character's integer label and frequency
    # print(f"char, index, count")
    # for i in range(num_letters):
    #     char_letter = char_to_int_encoder.inverse_transform([i, ])[0]
    #     print(repr(char_letter), i, all_freq[char_letter])

    return char_to_int_encoder, text_as_integers


if __name__ == "__main__":

    # text_path = "data/t8.shakespeare.txt"
    text_path = "data/sherlock.txt"

    with open(text_path, 'r') as f:
        raw_text = f.read()

    # load a cached LabelEncoder if it exists, otherwise create one
    root_name = pathlib.Path(text_path).stem
    try:
        with open(f"le_{root_name}.obj", "rb") as f:
            char_to_int_encoder = pickle.load(f)
        print("Loaded Cached Label Encoder")
        integer_input_data = char_to_int_encoder.transform([*raw_text])
        print("Converted Letters to Integers")
    except FileNotFoundError as e:
        print("Creating Label Encoder")
        char_to_int_encoder, integer_input_data = reduce_letters(raw_text)
        print("Converted Letters to Integers")
        with open(f"le_{root_name}.obj", "wb") as f:
            pickle.dump(char_to_int_encoder, f)
        print("Cached Label Encoder")

    scores = []
    discrete_encodings = []

    num_labels = len(char_to_int_encoder.classes_)

    # number of bits total for the encoder
    bits_per_discrete_state = 1
    # bits_per_discrete_state = 32
    num_s = num_labels * bits_per_discrete_state

    # history_window = 1
    history_window = 10
    num_statelets_per_column = 10
    # num_statelets_per_column = 30
    # num_statelets_per_column = 100

    d_thresh = 3


    discrete_encoder = DiscreteTransformer(num_v=num_labels, num_s=num_s)

    # for i in range(num_labels):
    #     discrete_encoder.set_value(i)
    #     discrete_encoder.feedforward()
    #     discrete_binary_array = np.array(discrete_encoder.output.bits, dtype=bool)
    #     discrete_encodings.append(discrete_binary_array)
    #
    #     char_letter = char_to_int_encoder.inverse_transform([i, ])[0]
    #     print(i, repr(char_letter))
    #     print(discrete_binary_array)

    print("DiscreteTransformer")
    print("num_labels, init statelets, allocated statelets")
    num_encoder_bits = len(discrete_encoder.output.bits)
    print(num_labels, num_s, num_encoder_bits)

    #num_rpd = 12,  # number of receptors per dendrite

    sl = ContextLearner(
            num_c=num_encoder_bits,  # number of columns
            num_spc=num_statelets_per_column,  # number of statelets per column
            num_dps=10,  # number of dendrites per statelet
            num_rpd=12,  # number of receptors per dendrite
            d_thresh=d_thresh,  # dendrite threshold
            perm_thr=20,  # receptor permanence threshold
            perm_inc=2,  # receptor permanence increment
            perm_dec=1,  # receptor permanence decrement
            num_t=1 + history_window,
            always_update=True)

    sl_bits = len(sl.output.bits)

    # hol = SequenceLearner(
    # num_c=num_encoder_bits,  # number of columns
    # num_spc=num_statelets_per_column,  # number of statelets per column
    # num_dps=10,  # number of dendrites per statelet
    # num_rpd=12,  # number of receptors per dendrite
    # d_thresh=6,  # dendrite threshold
    # perm_thr=20,  # receptor permanence threshold
    # perm_inc=2,  # receptor permanence increment
    # perm_dec=1,  # receptor permanence decrement
    # num_t=2,
    # always_update=True)
    #
    # hol_bits = len(hol.output.bits)

    # hol_pp = PatternPooler(
    # num_s=num_encoder_bits,  # number of statelets
    # num_as=8,  # number of active statelets
    # perm_thr=20,  # receptor permanence threshold
    # perm_inc=2,  # receptor permanence increment
    # perm_dec=1,  # receptor permanence decrement
    # pct_pool=0.8,  # pooling percentage
    # pct_conn=0.5,  # initially connected percentage
    # pct_learn=0.3)  # learn percentage

    # hol_pp_bits = len(hol_pp.output.bits)

    # print("Bits for each block output")
    # ic(sl_bits, hol_bits, hol_pp_bits)

    # # HOL, history 0, no PP, configuration
    # sl.input.add_child(discrete_encoder.output)
    # hol.input.add_child(sl.output, 0)
    # sl.context.add_child(hol.output, 0)

    # # Standard SequenceLearner configuration
    sl.input.add_child(discrete_encoder.output)
    for i in range(1, history_window + 1):
        sl.context.add_child(sl.output, i)

    # //for (uint32_t i = 1 ; i < num_t ; i++ ) {
    #     // Connect context to previous output
    # //    context.add_child(&output, i);
    # //}

    # # HOL, history 0, with PP, configuration
    # sl.input.add_child(discrete_encoder.output)
    # hol_pp.input.add_child(sl.output)
    # hol.input.add_child(hol_pp.output, 0)
    # sl.context.add_child(hol.output, 0)

    # # SL / concat HOL context, history 0, with PP, configuration
    # sl.input.add_child(discrete_encoder.output)
    # hol_pp.input.add_child(sl.output)
    # hol.input.add_child(hol_pp.output, 0)
    # sl.context.add_child(sl.output, 1)
    # sl.context.add_child(hol.output, 0)

    t0 = time()
    break_count = 5000

    # with (open("sl_hol_pp_match_encoder_bits_sherlock.csv", 'w') as f):
    # with (open("sl_hol_pp_match_encoder_bits_sherlock_scratch.csv", 'w') as f):
    # with (open("sl_sherlock_random.csv", 'w') as f):
    # with (open("sl_hol_pp_match_encoder_bits_sherlock_random.csv", 'w') as f):
    # with (open("sl_sherlock_thin_history_10_random.csv", 'w') as f):
    # with (open(f"sl_sherlock_wide_history_{history_window:02d}_random.csv", 'w') as f):
    output_csv = f"sl_sherlock_wide_{bits_per_discrete_state:02d}_deep_{num_statelets_per_column:03d}"
    output_csv += f"_history_{history_window:02d}_dthresh_{d_thresh:02d}_random.csv"
    # sl_sherlock_wide_01_deep_100_history_10_dthresh_3_random.cs
    with open(output_csv, "w") as f:

        # Loop through the values
        for i in range(len(integer_input_data)):
            # if i >= break_count:
            #     break

            # Set discrete transformer value
            discrete_encoder.set_value(integer_input_data[i])

            # Compute the discrete transformer
            discrete_encoder.feedforward()
            # discrete_binary_array = np.array(discrete_encoder.output.bits, dtype=bool)

            # Compute the sequence learner
            sl.feedforward(learn=True)

            sl_binary_array = np.array(sl.output.bits,
                                       dtype=bool)  # .reshape((num_encoder_bits, num_statelets_per_column))

            sl_context_array = np.array(sl.context.bits,
                                        dtype=bool)  # .reshape((num_encoder_bits, num_statelets_per_column))

            # print("Level 1")
            # for k in range(num_encoder_bits):
            #     if np.count_nonzero(sl_binary_array[k, :]) > 0:
            #         print(k, sl_binary_array[k, :])

            # Compute the pattern pooler
            # hol_pp.feedforward(learn=True)
            # hol_pp_binary_array = np.array(hol_pp.output.bits, dtype=bool)

            # Compute the higher-order learner
            # hol.feedforward(learn=True)

            # hol_binary_array = np.array(hol.output.bits, dtype=bool)#.reshape(
            # (num_encoder_bits, num_statelets_per_column))
            # print(len(sl_binary_array), len(sl_context_array), len(hol_binary_array), len(sl_binary_array)+len(hol_binary_array))

            # hol_binary_array = np.array(hol.output.bits, dtype=bool).reshape(
            #         (512, 10))
            # sl_binary_array = np.array(sl.output.bits, dtype=bool) #.reshape((num_encoder_bits, num_statelets_per_column))

            # print("Level 2")
            # for k in range(num_encoder_bits * num_statelets_per_column):
            #     if np.count_nonzero(hol_binary_array[k, :]) > 0:
            #         print(k, hol_binary_array[k, :])

            # print("second level output")
            # print(hol_binary_array)

            # Get anomaly score

            sl_nonzero = np.count_nonzero(sl_binary_array)
            # hol_pp_nonzero = np.count_nonzero(hol_pp_binary_array)
            # hol_nonzero = np.count_nonzero(hol_binary_array)

            sl_state_count = sl.get_historical_count()
            # hol_state_count = hol.get_historical_count()

            char_letter = char_to_int_encoder.inverse_transform([integer_input_data[i], ])[0]
            anom_score1 = sl.get_anomaly_score()
            # anom_score2 = hol.get_anomaly_score()
            scores.append(anom_score1)
            f.write(f"{i}, {repr(char_letter)}, {integer_input_data[i]}, {sl_state_count}, ")
            f.write(f"{sl_nonzero}, {anom_score1:.2f}\n")
            # f.write(f"{i}, {repr(char_letter)}, {integer_input_data[i]}, {sl_state_count}, {hol_state_count}, ")
            # f.write(f"{sl_nonzero}, {hol_nonzero}, {anom_score1:.2f}, {anom_score2:.2f}\n")

            # # Standard SequenceLearner configuration
            # sl_shakespeare.csv
            # 5458199 letters learned in 8372.40 sec. (1.53ms / letter)
            # sl_sherlock.csv
            # 581864 letters learned in 929.67 sec. (1.6ms / letter)

            # # HOL, history 0, no PP, configuration
            # hol_sherlock.csv
            # 581864 letters learned in 9210.79 sec. (15.8ms / letter)

            # # HOL, history 0, with PP, configuration
            # hol_pp_sherlock.csv
            # 5001 letters learned in 25.24 sec. (5ms / letter)

            # # SL / concat HOL context, history 0, with PP, configuration
            # sl_hol_pp_sherlock.csv
            # 5001 letters learned in 23.46 sec. (4.7ms / letter)
            # sl_hol_pp_sherlock_reduced_verbosity.csv
            # 5001 letters learned in 15.50 sec. (3.1ms / letter)

            # # SL / concat HOL context, match encoder bits, history 0, with PP, configuration
            # sl_hol_pp_match_encoder_bits_sherlock.csv
            # 5000 letters learned in 36.15 sec.

            # 5000 letters learned in 48.05 sec.
            # sl_hol_pp_match_encoder_bits_sherlock_scratch.csv

    t = time() - t0
    print(f"{break_count} letters learned in {t:.2f} sec.")
