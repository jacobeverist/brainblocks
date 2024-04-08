"""
Plot results of sequence learning experiments

"""

import matplotlib.pyplot as plt
import pandas as pd

# 581864 letters learned in 929.67 sec.
# print(i, integer_input_data[i], char_letter, np.count_nonzero(sl_binary_array), anom_score1, anom_score2)

# 581864 letters learned in 9210.79 sec.
# f.write(f"{i}, {repr(char_letter)}, {integer_input_data[i]}, {sl_nonzero}, {hol_nonzero}, {anom_score1:.2f}, {anom_score2:.2f}\n")

# 581838, 't', 77, 264, 0.00

# csv_files = ["sl_sherlock_deep_random.csv", "sl_sherlock_thin_history_05_random.csv",
#              "sl_sherlock_thin_history_10_random.csv", "sl_sherlock_wide_history_01_random.csv",
#              "sl_sherlock_wide_01_deep_100_history_10_dthresh_3_random.csv"]

csv_files = ["sl_sherlock_wide_01_deep_100_history_10_dthresh_3_random.csv",
             "sl_sherlock_wide_01_deep_010_history_10_dthresh_03_random.csv"]

for filename in csv_files:

    name_stem = filename[:-4]

    # f.write(f"{i}, {repr(char_letter)}, {integer_input_data[i]}, {sl_state_count}, ")
    #         f.write(f"{sl_nonzero}, {anom_score1:.2f}\n")

    column_names = ["step", "letter", "integer", "sl_states", "sl_acts", "sl_score"]
    df = pd.read_csv(filename, sep=", ", engine="python", names=column_names, index_col="step")
    df = df.drop(columns=['letter'])

    plt.clf()
    df_head = df[:1000]
    axes = df_head.plot(subplots=True, legend=False)
    for k in range(len(df_head.columns)):
        axes[k].set_ylabel(df_head.columns[k])
    plt.savefig(f"{name_stem}_head.png")
    plt.close()

    plt.clf()
    df_mid = df[225000:226000]
    axes = df_mid.plot(subplots=True, legend=False)
    for k in range(len(df_mid.columns)):
        axes[k].set_ylabel(df_mid.columns[k])
    plt.savefig(f"{name_stem}_mid.png")
    plt.close()

    plt.clf()
    df_tail = df[-1000:]
    axes = df_tail.plot(subplots=True, legend=False)
    for k in range(len(df_tail.columns)):
        axes[k].set_ylabel(df_tail.columns[k])
    plt.savefig(f"{name_stem}_tail.png")
    plt.close()

    # axes[1].set_ylim(0, 28480)
    # axes[2].set_ylim(0, 28480)

exit()

# header3 = ["step", "letter", "integer", "sl_acts", "hol_acts", "sl_score", "hol_score"]
# header3 = ["step", "letter", "integer", "sl_states", "sl_acts", "sl_score"]
header3 = ["step", "letter", "integer", "sl_states", "hol_states", "sl_acts", "hol_acts", "sl_score", "hol_score"]
df = pd.read_csv("sl_hol_pp_match_encoder_bits_sherlock_random.csv", sep=", ", engine="python", names=header3,
                 index_col="step")
# df = pd.read_csv("sl_sherlock_random.csv", sep=", ", engine="python", names=header3, index_col="step")
df = df.drop(columns=['letter'])

plt.clf()
df_head = df[:1000]
axes = df_head.plot(subplots=True, legend=False)
ic(axes)
for k in range(len(df_head.columns)):
    axes[k].set_ylabel(df_head.columns[k])
axes[1].set_ylim(0, 28480)
axes[2].set_ylim(0, 28480)
plt.savefig("sl_hol_pp_match_encoder_bits_sherlock_random.png")
plt.close()

exit()

header3 = ["step", "letter", "integer", "sl_acts", "hol_acts", "sl_score", "hol_score"]
df = pd.read_csv("sl_hol_pp_sherlock.csv", sep=", ", engine="python", names=header3, index_col="step")
df = df.drop(columns=['letter'])

plt.clf()
df_head = df[:1000]
axes = df_head.plot(subplots=True, legend=False)
ic(axes)
for k in range(len(df_head.columns)):
    axes[k].set_ylabel(df_head.columns[k])
plt.savefig("sl_hol_0_pp_sherlock_head.png")
plt.close()

exit()

header3 = ["step", "letter", "integer", "sl_acts", "hol_acts", "sl_score", "hol_score"]
df = pd.read_csv("hol_pp_sherlock.csv", sep=", ", engine="python", names=header3, index_col="step")
df = df.drop(columns=['letter'])

plt.clf()
df_head = df[:1000]
axes = df_head.plot(subplots=True, legend=False)
ic(axes)
for k in range(len(df_head.columns)):
    axes[k].set_ylabel(df_head.columns[k])
plt.savefig("hol_0_pp_sherlock_head.png")
plt.close()

exit()

header = ["step", "letter", "integer", "acts", "score1"]
df = pd.read_csv("sl_sherlock_out.txt", sep=", ", engine="python", names=header, index_col="step")
df = df.drop(columns=['letter'])

df_head = df[:1000]
ic(df_head)
axes = df_head.plot(subplots=True, legend=False)
ic(axes)
for k in range(len(df_head.columns)):
    axes[k].set_ylabel(df_head.columns[k])
plt.savefig("sl_sherlock_head.png")
plt.close()

# plt.clf()
# df_head = df[:1000]
# plt.plot(df_head.step, df_head.score1)
# plt.savefig("sl_sherlock_head.png")

# plt.clf()
# df_tail = df[-1000:]
# plt.plot(df_tail.step, df_tail.score1)
# plt.show()

header2 = ["step", "letter", "integer", "sl_acts", "hol_acts", "sl_score", "hol_score"]
df = pd.read_csv("sl2_sherlock_out.txt", sep=", ", engine="python", names=header2, index_col="step")
df = df.drop(columns=['letter'])

plt.clf()
df_head = df[:1000]
axes = df_head.plot(subplots=True, legend=False)
ic(axes)
for k in range(len(df_head.columns)):
    axes[k].set_ylabel(df_head.columns[k])
plt.savefig("hol_1_sherlock_head.png")
plt.close()
# plt.plot(df_head.step, df_head.sl_score, df_head.hol_score)

# plt.clf()
# df_tail = df[-1000:]
# plt.plot(df_tail.step, df_tail.sl_score, df_tail.hol_score)
# plt.savefig("sl2_sherlock_head.png")

header3 = ["step", "letter", "integer", "sl_acts", "hol_acts", "sl_score", "hol_score"]
df = pd.read_csv("hol_sherlock.csv", sep=", ", engine="python", names=header3, index_col="step")
df = df.drop(columns=['letter'])

plt.clf()
df_head = df[:1000]
axes = df_head.plot(subplots=True, legend=False)
ic(axes)
for k in range(len(df_head.columns)):
    axes[k].set_ylabel(df_head.columns[k])
plt.savefig("hol_0_sherlock_head.png")
plt.close()

# plt.clf()
# plt.plot(df.step, df.sl_score, df.hol_score)
# plt.savefig("hol_0_sherlock_head.png")


# generate plot of data
# print("Saving to " + output_name + ".png")
# axes = sigDF.plot(subplots=True, legend=False)
# for k in range(len(sigDF.columns)):
#     axes[k].set_ylabel(sigDF.columns[k])
# plt.savefig(output_name + ".png")
# plt.close()
