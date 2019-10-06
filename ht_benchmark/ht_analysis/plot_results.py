#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot results of the high-throughput benchmark
"""
__author__ = "Miguel Hernández Cabronero <mhernandez@deic.uab.cat>"
__date__ = "10/09/2019"

import os
import matplotlib.pyplot as plt
import pandas as pd
import collections
import math
import itertools

from plotdata import ScatterData

# -------------------------- Begin configurable part

input_csv_path = "withfapec_benchmark_results.csv"
if not os.path.exists(input_csv_path):
    input_csv_path = "benchmark_results.csv"
    if not os.path.exists(input_csv_path):
        raise RuntimeError(f"{input_csv_path} does not exist."
                           f"Run /bin/benchmark and copy the resulting .csv there, or run /main.sh directly.")
    if __name__ == '__main__':
        print("FAPEC results not found - try running ./compress_fapec.py")
entropy_csv_path = "entropy_results.csv"
output_tex_path_template = "compression_decompression_{}.tex"
speed_factor = 10**9
marker_size = 50

target_table_columns = ["compression_efficiency", "compression_speed_mss", "decompression_speed_mss"]

table_label_group_list = [
    ["ISO", ["iso_12640_2"]],
    ["Kodak", ["kodak_photocd"]],
    ["Rawzor", ["rawzor"]],
    ["Mixed", ["mixed_datasets"]]]
table_label_group_list.append(["All", set(itertools.chain(*(g for _, g in table_label_group_list)))])
included_dirs = table_label_group_list[-1][1]

# Be verbose?
be_verbose = True


# -------------------------- End configurable part

def entropy(data):
    counts = collections.Counter(data).values()
    assert sum(counts) == len(data)
    probabilities = (c / len(data) for c in counts)
    return - sum(p * math.log2(p) for p in probabilities)


def get_image_entropy_df(image_paths):
    if os.path.exists(entropy_csv_path):
        entropy_df = pd.read_csv(entropy_csv_path)
        entropy_df = entropy_df.set_index("image_path", drop=False)
    else:
        entropy_df = pd.DataFrame(columns=["image_path", "width", "height", "bitdepth", "entropy"])
        entropy_df = entropy_df.set_index("image_path", drop=False)

    for image_path in image_paths:
        if image_path not in entropy_df["image_path"]:
            entropy_df.loc[image_path] = pd.Series(dict(
                image_path=image_path, width=os.path.getsize(image_path), height=1,
                bitdepth=8, entropy=entropy(open(image_path, "rb").read())))
            print(f"Added entropy for {image_path}")

    entropy_df.to_csv(entropy_csv_path, index=False)
    return entropy_df

def export_legend(legend, filename):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)

if __name__ == '__main__':
    full_df = pd.read_csv(input_csv_path)
    full_df = full_df[full_df["lossless"] == 1] # No surprises
    full_df = pd.concat([full_df[full_df.directory.str.contains(d)]
                         for d in included_dirs])

    # original_data_df = full_df[full_df["directory"].str.contains("dpcm")]
    dpcm_data_df = full_df[full_df["directory"].str.contains("dpcm")]
    original_data_df = full_df[~full_df["directory"].str.contains("dpcm")]

    pretty_dict = {
        "Marlin2019 distType:Laplacian K:10 O:0 minMarlinSymbols:2 numDict:11 purgeProbabilityThreshold:4.76837e-07 K:10 O:0 S:1.72727":
            r"Algorithm 6 :: $|\mathcal{F}\| = 2^O = 1$ :: $|\mathcal{T}\ | = 1024$",
        "Marlin2019 distType:Laplacian K:10 O:2 minMarlinSymbols:2 numDict:11 purgeProbabilityThreshold:4.76837e-07 K:10 O:2 S:1.90909":
            r"Algorithm 6 :: $|\mathcal{F}\| = 2^O = 4$ :: $|\mathcal{T}\ | = 1024$",
        "Marlin2019 distType:Laplacian K:8 O:0 minMarlinSymbols:2 numDict:11 purgeProbabilityThreshold:4.76837e-07 K:8 O:0 S:1.90909":
            r"Algorithm 6 :: $|\mathcal{F}\| = 2^O = 1$ :: $|\mathcal{T}\ | = 256$",
        "Marlin2019 distType:Laplacian K:8 O:2 minMarlinSymbols:2 numDict:11 purgeProbabilityThreshold:4.76837e-07 K:8 O:2 S:1.90909":
            r"Algorithm 6 :: $|\mathcal{F}\| = 2^O = 4$ :: $|\mathcal{T}\ | = 256$",
        "fapec": "FAPEC",
        "Zstd1": "Zstd",
        "Lzo1-15 1": "LZO",
        "Lz4": "LZ4",
        "iso_12640_1": "ISO 12640-1",
        "iso_12640_2": "ISO 12640-2",
        "iso_ccitt": "ISO CCITT",
        "kodak_photocd": "KODAK",
        "rawzor": "Rawzor",
        # "uci_datasets": "UCI Mixed",
        "mixed_datasets": "Mixed",
        "compression_efficiency": r"Compression efficiency $\eta_\aleph$",
        "decompression_speed_mss": r"Decompression speed ($10^{9} \cdot$ samples / s)",
        "compression_speed_mss": r"Compression speed ($10^{9} \cdot$ samples / s)",
    }

    markers = ["X", "^", "P", "s", "p", "<", ">"]
    colors = ["r", "g", "b", "orange", "black"]

    for df_label, df in [("original_images", original_data_df)]:
        # Set properties table
        entropy_df = get_image_entropy_df(df["file"].unique())
        entropy_df["directory"] = entropy_df.image_path.apply(lambda p: os.path.basename(os.path.dirname(p)))
        entropy_df["sample_count"] = entropy_df["width"] * entropy_df["height"]
        print(r"\begin{tabular}{l c c}")
        print(r"""\toprule
\textbf{Directory} & \textbf{$\mathbf{10^6} \cdot$~Samples} & \textbf{Entropy (bps)}\\
\toprule""")
        df_by_dir = {}
        for dir, dir_df in entropy_df.groupby("directory"):
            df_by_dir[dir] = dir_df
            if any(s in dir for s in ("dpcm", "uci_")):
                continue
            fields = [pretty_dict[dir] if dir in pretty_dict else dir]
            fields.append(f"{dir_df['sample_count'].sum() / 10**6:.2f}")
            # fields.append(f"{dir_df['entropy'].min():.2f}")
            # fields.append(f"{dir_df['entropy'].max():.2f}")
            fields.append(f"{dir_df['entropy'].mean():.2f}")
            print(" & ".join(fields) + r"\\")
        print("")
        for group_label, dir_list in table_label_group_list:
            group_df = pd.concat([df_by_dir[dir] for dir in dir_list])
            fields = [pretty_dict[group_label] if group_label in pretty_dict else group_label]
            fields.append(f"{group_df['sample_count'].sum() / 10 ** 6:.2f}")
            # fields.append(f"{group_df['entropy'].min():.2f}")
            # fields.append(f"{group_df['entropy'].max():.2f}")
            fields.append(f"{group_df['entropy'].mean():.2f}")
            print(" & ".join(fields) + r"\\")
        print(r"\bottomrule")
        print(r"\end{tabular}")

        df["image_path"] = df["file"]
        image_paths = df["image_path"].unique()
        df = df.join(get_image_entropy_df(image_paths).set_index("image_path"), on="image_path")
        df["compression_rate_bps"] = (df["compression_bytes"] * 8) / df["pixel_count"]
        df["compression_efficiency"] = df["entropy"] / df["compression_rate_bps"]
        df["compression_speed_mss"] = (df["pixel_count"] / df["compression_avg_time_s"]) / speed_factor
        df["decompression_speed_mss"] = (df["pixel_count"] / df["decompression_avg_time_s"]) / speed_factor

        # Compression time vs bps
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # x_dimension = "compression_rate_bps"
        x_dimension = "compression_efficiency"
        y_dimension = "compression_speed_mss"
        plt_subirs = False
        all_codec_data = []
        for i, (codec_name, codec_df) in enumerate(df.groupby("codec_name")):
            codec_data = ScatterData(x_values=[codec_df[x_dimension].mean()],
                                     y_values=[codec_df[y_dimension].mean()])
            codec_data.label = codec_name if codec_name not in pretty_dict else pretty_dict[codec_name]
            codec_data.x_label = pretty_dict[x_dimension]
            codec_data.y_label =  pretty_dict[y_dimension]
            codec_data.extra_kwargs = dict(
                marker=markers[i % len(markers)] if "marlin" not in codec_name.lower() else "o",
                color=colors[i % len(colors)],
                s=marker_size)
            all_codec_data.append(codec_data)
        for data in sorted(all_codec_data, key=lambda pd: pd.label):
            # leg, data.label = data.label, None
            data.render()
            # data.label = leg
        legend = plt.legend(loc="lower center", bbox_to_anchor=(0.5, 1), ncol=4)
        export_legend(legend, "legend_compression.pdf")
        # plt.xlim(4.5, 8.5)
        # plt.xlim(0, 1)
        # plt.ylim(0, 7)
        plt.savefig(f"allcodecs_{df_label}_coding_speed_vs_compression_rate.pdf", bbox_inches="tight")
        plt.close()

        # Decompression time vs bps
        fig = plt.figure()
        ax = fig.add_subplot(111)
        x_dimension = "compression_efficiency"
        y_dimension = "decompression_speed_mss"
        plt_subirs = False
        all_codec_data = []
        for i, (codec_name, codec_df) in enumerate(df.groupby("codec_name")):
            codec_data = ScatterData(x_values=[codec_df[x_dimension].mean()],
                                     y_values=[codec_df[y_dimension].mean()])
            codec_data.label = codec_name if codec_name not in pretty_dict else pretty_dict[codec_name]
            codec_data.x_label = pretty_dict[x_dimension]
            codec_data.y_label = pretty_dict[y_dimension]
            codec_data.extra_kwargs = dict(
                marker=markers[i % len(markers)] if "marlin" not in codec_name.lower() else "o",
                color=colors[i % len(colors)],
                s=marker_size)
            all_codec_data.append(codec_data)
        for data in sorted(all_codec_data, key=lambda pd: pd.label):
            # leg, data.label = data.label, None
            data.render()
            # data.label = leg
        plt.legend(loc="lower center", bbox_to_anchor=(0.5, 1), ncol=4)
        # plt.xlim(4.5, 8.5)
        # plt.xlim(0, 1)
        # plt.ylim(0, 7)
        plt.savefig(f"allcodecs_{df_label}_decoding_speed_vs_compression_rate.pdf", bbox_inches="tight")
        plt.close()


        if df_label != "original_images":
            continue
        compression_table_csv_file = open(output_tex_path_template.format(df_label), "w")

        target_dirs = sorted(os.path.basename(d) for d in df["directory"].unique())
        compression_table_csv_file.write(r"\toprule" + "\n")
        compression_table_csv_file.write(r" &")
        compression_table_csv_file.write(
            " & ".join(r"\textbf{" + group_label + "}"  for group_label, _ in table_label_group_list))
        compression_table_csv_file.write(r"\\" + "\n")
        compression_table_csv_file.write(r"\textbf{Codec} & ")
        compression_table_csv_file.write(" & ".join([r"$\efficiencySymbol$\hspace{0.5cm}$\coder_s$\hspace{0.6cm}$\decoder_s$"] * len(table_label_group_list)))
        compression_table_csv_file.write(r"\\" + "\n")
        compression_table_csv_file.write(r"\toprule" + "\n")


        def filter_label(l):
            for original, replacement in {
                "Marlin2019 distType:Laplacian K:10 O:0 minMarlinSymbols:2 numDict:11 purgeProbabilityThreshold:4.76837e-07 K:10 O:0 S:1.72727":
                    r"Alg.~\ref{alg:markov_forest}~--~K:10, O:0",

                "Marlin2019 distType:Laplacian K:10 O:2 minMarlinSymbols:2 numDict:11 purgeProbabilityThreshold:4.76837e-07 K:10 O:2 S:1.90909":
                    r"Alg.~\ref{alg:markov_forest}~--~K:10, O:2",

                "Marlin2019 distType:Laplacian K:8 O:0 minMarlinSymbols:2 numDict:11 purgeProbabilityThreshold:4.76837e-07 K:8 O:0 S:1.90909":
                    r"Alg.~\ref{alg:markov_forest}~--~K:8, O:0",

                "Marlin2019 distType:Laplacian K:8 O:2 minMarlinSymbols:2 numDict:11 purgeProbabilityThreshold:4.76837e-07 K:8 O:2 S:1.90909":
                    r"Alg.~\ref{alg:markov_forest}~--~K:8, O:2"}.items():
                l = l.replace(original, replacement)
            try:
                return pretty_dict[l]
            except KeyError:
                return l
        codecdf_by_codecname = {filter_label(os.path.basename(label)): df
                                for label, df in df.groupby("codec_name")}
        codecs = sorted(codecdf_by_codecname.keys())
        for codec in codecs:
            compression_table_csv_file.write(f"{codec} & ")
            fields = []
            # Per group
            codec_df = codecdf_by_codecname[codec]
            for group_label, dir_list in table_label_group_list:
                group_df = pd.concat([codec_df[codec_df.directory.str.contains(dir)]
                                       for dir in dir_list])
                rate, c_time, d_time = group_df[target_table_columns].mean()
                fields.append(f"{rate:.2f}~{c_time:.3f}~{d_time:.4f}")
            # # Grand total
            # rate, c_time, d_time = codec_df[
            #     target_table_columns].mean()
            # fields.append(f"{rate:.2f}~{c_time:.3f}~{d_time:.4f}")
            # Out
            compression_table_csv_file.write(" & ".join(fields))
            compression_table_csv_file.write(r"\\" + "\n")
        compression_table_csv_file.write(r"\bottomrule" + "\n")
