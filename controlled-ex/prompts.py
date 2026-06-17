prompts = [prompt_1, prompt_2, prompt_3, prompt_4, prompt_5, prompt_6, prompt_7, prompt_8, prompt_9, prompt_10]

prompt_1='''You are selecting clients for federated continual learning.
A good client subset should:
1. Cover the currently seen classes as evenly as possible.
2. Include clients containing recently introduced or frontier classes so the model adapts to new concepts.
3. Include some clients containing previously seen classes to reduce forgetting.
4. Avoid selecting many clients with the same dominant labels.
5. Cover diverse corruption conditions and severity levels.
6. Prefer clients with positive local learning progress.
7. Avoid selecting too many unstable clients with extreme update norms or conflicting update directions.
You must output strict JSON only.
Output schema:
{
  "selected_clients": [0, 1, 2],
  "reason": "short explanation",
  "selection_summary": {
    "class_coverage": "short text",
    "drift_diversity": "short text",
    "stability": "short text"
  }
}

Input:
{
  "candidate_client_reports": [
    {
      "cid": 9,
      "class_coverage_ratio": 0.65,
      "data_drift": {
        "corruption_type": "gaussian_blur",
        "severity": 4
      },
      "label_entropy": 1.6494897617737998,
      "label_histogram": {
        "1": 17,
        "11": 110,
        "12": 11,
        "13": 6,
        "15": 1,
        "16": 1,
        "18": 1,
        "19": 146,
        "3": 2,
        "5": 64,
        "7": 195,
        "8": 7,
        "9": 3
      },
      "local_eval": {
        "local_val_acc": 0.4,
        "loss_after": 1.8701288938522338,
        "loss_before": 5.318638515472412
      },
      "num_samples": 564,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 61.86749355789698
      }
    },
    {
      "cid": 10,
      "class_coverage_ratio": 0.55,
      "data_drift": {
        "corruption_type": "brightness",
        "severity": 1
      },
      "label_entropy": 1.7228936791234393,
      "label_histogram": {
        "0": 2,
        "10": 107,
        "11": 15,
        "12": 1,
        "13": 2,
        "15": 38,
        "16": 7,
        "2": 27,
        "4": 9,
        "8": 64,
        "9": 4
      },
      "local_eval": {
        "local_val_acc": 0.3188405797101449,
        "loss_after": 3.888459025949672,
        "loss_before": 5.257673415584841
      },
      "num_samples": 276,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 31.5467610155322
      }
    },
    {
      "cid": 13,
      "class_coverage_ratio": 0.35,
      "data_drift": {
        "corruption_type": "motion_blur",
        "severity": 5
      },
      "label_entropy": 1.0878670854864663,
      "label_histogram": {
        "11": 11,
        "14": 6,
        "17": 251,
        "18": 2,
        "2": 52,
        "6": 17,
        "8": 31
      },
      "local_eval": {
        "local_val_acc": 0.6630434782608695,
        "loss_after": 1.7899275344351064,
        "loss_before": 5.275343708370043
      },
      "num_samples": 370,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 41.68719179600916
      }
    },
    {
      "cid": 11,
      "class_coverage_ratio": 0.7,
      "data_drift": {
        "corruption_type": "motion_blur",
        "severity": 3
      },
      "label_entropy": 2.131498794507011,
      "label_histogram": {
        "0": 1,
        "1": 2,
        "10": 5,
        "11": 1,
        "12": 93,
        "15": 73,
        "16": 100,
        "17": 57,
        "4": 62,
        "5": 5,
        "6": 13,
        "7": 13,
        "8": 30,
        "9": 38
      },
      "local_eval": {
        "local_val_acc": 0.2926829268292683,
        "loss_after": 2.819660041390396,
        "loss_before": 5.336191890685539
      },
      "num_samples": 493,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 54.89919894054722
      }
    },
    {
      "cid": 14,
      "class_coverage_ratio": 0.65,
      "data_drift": {
        "corruption_type": "contrast",
        "severity": 3
      },
      "label_entropy": 1.6982446330931011,
      "label_histogram": {
        "0": 3,
        "10": 7,
        "11": 7,
        "12": 214,
        "14": 75,
        "15": 14,
        "16": 1,
        "18": 67,
        "19": 3,
        "2": 128,
        "4": 1,
        "5": 1,
        "7": 35
      },
      "local_eval": {
        "local_val_acc": 0.6376811594202898,
        "loss_after": 1.3661204369171807,
        "loss_before": 5.329573645107988
      },
      "num_samples": 556,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 61.61393419946819
      }
    },
    {
      "cid": 3,
      "class_coverage_ratio": 0.5,
      "data_drift": {
        "corruption_type": "contrast",
        "severity": 1
      },
      "label_entropy": 1.0842247859989427,
      "label_histogram": {
        "1": 1,
        "10": 12,
        "11": 68,
        "12": 1,
        "14": 1,
        "16": 1,
        "2": 1,
        "4": 96,
        "5": 1,
        "8": 1
      },
      "local_eval": {
        "local_val_acc": 0.5333333333333333,
        "loss_after": 4.491059398651123,
        "loss_before": 5.326913780636257
      },
      "num_samples": 183,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 21.24681248395326
      }
    },
    {
      "cid": 6,
      "class_coverage_ratio": 0.55,
      "data_drift": {
        "corruption_type": "pixelate",
        "severity": 1
      },
      "label_entropy": 1.1818091538526883,
      "label_histogram": {
        "0": 48,
        "1": 9,
        "11": 1,
        "13": 1,
        "14": 1,
        "15": 1,
        "17": 59,
        "3": 8,
        "4": 1,
        "5": 260,
        "9": 16
      },
      "local_eval": {
        "local_val_acc": 0.6534653465346535,
        "loss_after": 1.854526177491292,
        "loss_before": 5.288882822093397
      },
      "num_samples": 405,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 45.04923790267053
      }
    },
    {
      "cid": 12,
      "class_coverage_ratio": 0.5,
      "data_drift": {
        "corruption_type": "jpeg_compression",
        "severity": 4
      },
      "label_entropy": 1.9295253971294104,
      "label_histogram": {
        "20": 31,
        "21": 7,
        "22": 3,
        "23": 59,
        "25": 17,
        "26": 11,
        "27": 5,
        "28": 16,
        "31": 26,
        "36": 3
      },
      "local_eval": {
        "local_val_acc": 0.45454545454545453,
        "loss_after": 4.813415657390248,
        "loss_before": 5.33248350837014
      },
      "num_samples": 178,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 21.105555043991597
      }
    },
    {
      "cid": 8,
      "class_coverage_ratio": 0.45,
      "data_drift": {
        "corruption_type": "contrast",
        "severity": 2
      },
      "label_entropy": 1.2445300896481535,
      "label_histogram": {
        "1": 251,
        "10": 2,
        "17": 3,
        "18": 93,
        "2": 2,
        "3": 1,
        "4": 72,
        "6": 19,
        "9": 8
      },
      "local_eval": {
        "local_val_acc": 0.6785714285714286,
        "loss_after": 1.6243350676127843,
        "loss_before": 5.359443323952811
      },
      "num_samples": 451,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 51.96731256879458
      }
    },
    {
      "cid": 18,
      "class_coverage_ratio": 0.45,
      "data_drift": {
        "corruption_type": "brightness",
        "severity": 1
      },
      "label_entropy": 1.547003479217219,
      "label_histogram": {
        "1": 3,
        "12": 2,
        "16": 4,
        "18": 30,
        "3": 45,
        "4": 3,
        "7": 72,
        "8": 14,
        "9": 2
      },
      "local_eval": {
        "local_val_acc": 0.3488372093023256,
        "loss_after": 4.743556299874949,
        "loss_before": 5.324154133020445
      },
      "num_samples": 175,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 21.290141663287944
      }
    },
    {
      "cid": 0,
      "class_coverage_ratio": 0.6,
      "data_drift": {
        "corruption_type": "clean",
        "severity": 0
      },
      "label_entropy": 1.2372041448349067,
      "label_histogram": {
        "1": 1,
        "11": 1,
        "13": 167,
        "14": 1,
        "16": 55,
        "18": 2,
        "2": 4,
        "4": 6,
        "5": 3,
        "6": 2,
        "8": 2,
        "9": 32
      },
      "local_eval": {
        "local_val_acc": 0.6666666666666666,
        "loss_after": 2.5790224524511807,
        "loss_before": 5.292057154835135
      },
      "num_samples": 276,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 31.351796161621063
      }
    },
    {
      "cid": 5,
      "class_coverage_ratio": 0.3,
      "data_drift": {
        "corruption_type": "brightness",
        "severity": 2
      },
      "label_entropy": 1.172803582731704,
      "label_histogram": {
        "11": 53,
        "15": 1,
        "2": 1,
        "4": 24,
        "5": 13,
        "8": 4
      },
      "local_eval": {
        "local_val_acc": 0.375,
        "loss_after": 5.178791046142578,
        "loss_before": 5.301027774810791
      },
      "num_samples": 96,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 10.330462003616006
      }
    },
    {
      "cid": 15,
      "class_coverage_ratio": 0.55,
      "data_drift": {
        "corruption_type": "pixelate",
        "severity": 2
      },
      "label_entropy": 1.5472977291225964,
      "label_histogram": {
        "0": 2,
        "1": 1,
        "10": 11,
        "11": 16,
        "13": 222,
        "14": 298,
        "15": 30,
        "17": 27,
        "19": 226,
        "4": 1,
        "8": 21
      },
      "local_eval": {
        "local_val_acc": 0.5305164319248826,
        "loss_after": 1.9506576855977376,
        "loss_before": 5.302792184229748
      },
      "num_samples": 855,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 92.67687982348824
      }
    },
    {
      "cid": 19,
      "class_coverage_ratio": 0.6,
      "data_drift": {
        "corruption_type": "jpeg_compression",
        "severity": 3
      },
      "label_entropy": 1.6848773817720633,
      "label_histogram": {
        "0": 7,
        "1": 51,
        "10": 1,
        "12": 1,
        "14": 2,
        "15": 144,
        "3": 2,
        "4": 5,
        "5": 34,
        "7": 28,
        "8": 91,
        "9": 6
      },
      "local_eval": {
        "local_val_acc": 0.41935483870967744,
        "loss_after": 2.606143105414606,
        "loss_before": 5.30544848595896
      },
      "num_samples": 372,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 41.58923944162744
      }
    },
    {
      "cid": 16,
      "class_coverage_ratio": 0.55,
      "data_drift": {
        "corruption_type": "clean",
        "severity": 0
      },
      "label_entropy": 1.6084735623994935,
      "label_histogram": {
        "1": 9,
        "10": 212,
        "13": 1,
        "14": 7,
        "15": 19,
        "2": 1,
        "3": 19,
        "4": 1,
        "6": 106,
        "8": 134,
        "9": 39
      },
      "local_eval": {
        "local_val_acc": 0.22627737226277372,
        "loss_after": 2.376430657658264,
        "loss_before": 5.265924596438443
      },
      "num_samples": 548,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 61.88517497139714
      }
    },
    {
      "cid": 2,
      "class_coverage_ratio": 0.6,
      "data_drift": {
        "corruption_type": "jpeg_compression",
        "severity": 3
      },
      "label_entropy": 1.7705474114506652,
      "label_histogram": {
        "1": 1,
        "10": 18,
        "11": 38,
        "13": 3,
        "14": 1,
        "18": 3,
        "19": 10,
        "2": 19,
        "4": 1,
        "5": 5,
        "7": 59,
        "9": 1
      },
      "local_eval": {
        "local_val_acc": 0.46153846153846156,
        "loss_after": 4.980316590040158,
        "loss_before": 5.323369723099929
      },
      "num_samples": 159,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 17.57749106229022
      }
    },
    {
      "cid": 4,
      "class_coverage_ratio": 0.35,
      "data_drift": {
        "corruption_type": "jpeg_compression",
        "severity": 1
      },
      "label_entropy": 1.253357109098049,
      "label_histogram": {
        "0": 2,
        "1": 48,
        "10": 1,
        "16": 3,
        "18": 35,
        "19": 13,
        "9": 1
      },
      "local_eval": {
        "local_val_acc": 0.52,
        "loss_after": 4.957350730895996,
        "loss_before": 5.372942924499512
      },
      "num_samples": 103,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 13.964551739061749
      }
    },
    {
      "cid": 17,
      "class_coverage_ratio": 0.55,
      "data_drift": {
        "corruption_type": "gaussian_blur",
        "severity": 5
      },
      "label_entropy": 1.3525853257191967,
      "label_histogram": {
        "0": 324,
        "1": 1,
        "10": 1,
        "11": 78,
        "12": 1,
        "14": 8,
        "15": 56,
        "16": 12,
        "17": 1,
        "2": 155,
        "7": 3
      },
      "local_eval": {
        "local_val_acc": 0.660377358490566,
        "loss_after": 1.7063026278273865,
        "loss_before": 5.338122547797437
      },
      "num_samples": 640,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 68.56024413218987
      }
    },
    {
      "cid": 1,
      "class_coverage_ratio": 0.6,
      "data_drift": {
        "corruption_type": "pixelate",
        "severity": 3
      },
      "label_entropy": 1.5962783847217634,
      "label_histogram": {
        "0": 13,
        "10": 1,
        "11": 3,
        "14": 2,
        "15": 27,
        "16": 14,
        "17": 1,
        "18": 163,
        "19": 1,
        "4": 115,
        "5": 13,
        "9": 153
      },
      "local_eval": {
        "local_val_acc": 0.4365079365079365,
        "loss_after": 1.7167714001640442,
        "loss_before": 5.323762780144101
      },
      "num_samples": 506,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 55.15930760773831
      }
    },
    {
      "cid": 7,
      "class_coverage_ratio": 0.6,
      "data_drift": {
        "corruption_type": "shot_noise",
        "severity": 4
      },
      "label_entropy": 1.4361544730284694,
      "label_histogram": {
        "0": 1,
        "10": 24,
        "11": 1,
        "13": 1,
        "15": 1,
        "16": 63,
        "17": 3,
        "2": 7,
        "3": 315,
        "4": 1,
        "6": 174,
        "9": 98
      },
      "local_eval": {
        "local_val_acc": 0.46511627906976744,
        "loss_after": 1.6862897235293721,
        "loss_before": 5.3245672070702845
      },
      "num_samples": 689,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 75.6085097383853
      }
    }
  ],
  "clients_per_round": 5,
  "global_task": 1,
  "round_id": 1,
  "seen_classes": [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19
  ]
}'''
prompt_2='''You are selecting clients for federated continual learning.
A good client subset should:
1. Cover the currently seen classes as evenly as possible.
2. Include clients containing recently introduced or frontier classes so the model adapts to new concepts.
3. Include some clients containing previously seen classes to reduce forgetting.
4. Avoid selecting many clients with the same dominant labels.
5. Cover diverse corruption conditions and severity levels.
6. Prefer clients with positive local learning progress.
7. Avoid selecting too many unstable clients with extreme update norms or conflicting update directions.
You must output strict JSON only.
Output schema:
{
  "selected_clients": [0, 1, 2],
  "reason": "short explanation",
  "selection_summary": {
    "class_coverage": "short text",
    "drift_diversity": "short text",
    "stability": "short text"
  }
}

Input:
{
  "candidate_client_reports": [
    {
      "cid": 2,
      "class_coverage_ratio": 0.6,
      "data_drift": {
        "corruption_type": "jpeg_compression",
        "severity": 3
      },
      "label_entropy": 1.7705474114506652,
      "label_histogram": {
        "1": 1,
        "10": 18,
        "11": 38,
        "13": 3,
        "14": 1,
        "18": 3,
        "19": 10,
        "2": 19,
        "4": 1,
        "5": 5,
        "7": 59,
        "9": 1
      },
      "local_eval": {
        "local_val_acc": 0.3076923076923077,
        "loss_after": 3.6726851952381625,
        "loss_before": 4.769742012023926
      },
      "num_samples": 159,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 15.691496923641452
      }
    },
    {
      "cid": 6,
      "class_coverage_ratio": 0.55,
      "data_drift": {
        "corruption_type": "pixelate",
        "severity": 1
      },
      "label_entropy": 1.1818091538526883,
      "label_histogram": {
        "0": 48,
        "1": 9,
        "11": 1,
        "13": 1,
        "14": 1,
        "15": 1,
        "17": 59,
        "3": 8,
        "4": 1,
        "5": 260,
        "9": 16
      },
      "local_eval": {
        "local_val_acc": 0.6633663366336634,
        "loss_after": 2.7392492577581122,
        "loss_before": 4.493521671484013
      },
      "num_samples": 405,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 43.430021792257435
      }
    },
    {
      "cid": 11,
      "class_coverage_ratio": 0.7,
      "data_drift": {
        "corruption_type": "motion_blur",
        "severity": 3
      },
      "label_entropy": 2.131498794507011,
      "label_histogram": {
        "0": 1,
        "1": 2,
        "10": 5,
        "11": 1,
        "12": 93,
        "15": 73,
        "16": 100,
        "17": 57,
        "4": 62,
        "5": 5,
        "6": 13,
        "7": 13,
        "8": 30,
        "9": 38
      },
      "local_eval": {
        "local_val_acc": 0.4634146341463415,
        "loss_after": 2.102648390987055,
        "loss_before": 3.974068226853037
      },
      "num_samples": 493,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 53.78221496343634
      }
    },
    {
      "cid": 8,
      "class_coverage_ratio": 0.45,
      "data_drift": {
        "corruption_type": "contrast",
        "severity": 2
      },
      "label_entropy": 1.2445300896481535,
      "label_histogram": {
        "1": 251,
        "10": 2,
        "17": 3,
        "18": 93,
        "2": 2,
        "3": 1,
        "4": 72,
        "6": 19,
        "9": 8
      },
      "local_eval": {
        "local_val_acc": 0.6607142857142857,
        "loss_after": 1.8479687316077096,
        "loss_before": 4.812716552189419
      },
      "num_samples": 451,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 50.5818582733419
      }
    },
    {
      "cid": 13,
      "class_coverage_ratio": 0.35,
      "data_drift": {
        "corruption_type": "motion_blur",
        "severity": 5
      },
      "label_entropy": 1.0878670854864663,
      "label_histogram": {
        "11": 11,
        "14": 6,
        "17": 251,
        "18": 2,
        "2": 52,
        "6": 17,
        "8": 31
      },
      "local_eval": {
        "local_val_acc": 0.6630434782608695,
        "loss_after": 1.819011133650075,
        "loss_before": 4.827056366464366
      },
      "num_samples": 370,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 39.99177596760969
      }
    },
    {
      "cid": 12,
      "class_coverage_ratio": 0.5,
      "data_drift": {
        "corruption_type": "jpeg_compression",
        "severity": 4
      },
      "label_entropy": 1.9295253971294104,
      "label_histogram": {
        "20": 31,
        "21": 7,
        "22": 3,
        "23": 59,
        "25": 17,
        "26": 11,
        "27": 5,
        "28": 16,
        "31": 26,
        "36": 3
      },
      "local_eval": {
        "local_val_acc": 0.022727272727272728,
        "loss_after": 4.233046921816739,
        "loss_before": 5.41296876560558
      },
      "num_samples": 178,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 19.158166658411492
      }
    },
    {
      "cid": 15,
      "class_coverage_ratio": 0.55,
      "data_drift": {
        "corruption_type": "pixelate",
        "severity": 2
      },
      "label_entropy": 1.5472977291225964,
      "label_histogram": {
        "0": 2,
        "1": 1,
        "10": 11,
        "11": 16,
        "13": 222,
        "14": 298,
        "15": 30,
        "17": 27,
        "19": 226,
        "4": 1,
        "8": 21
      },
      "local_eval": {
        "local_val_acc": 0.5821596244131455,
        "loss_after": 1.9506633572735137,
        "loss_before": 4.9971149538604305
      },
      "num_samples": 855,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 91.9618858645086
      }
    },
    {
      "cid": 1,
      "class_coverage_ratio": 0.6,
      "data_drift": {
        "corruption_type": "pixelate",
        "severity": 3
      },
      "label_entropy": 1.5962783847217634,
      "label_histogram": {
        "0": 13,
        "10": 1,
        "11": 3,
        "14": 2,
        "15": 27,
        "16": 14,
        "17": 1,
        "18": 163,
        "19": 1,
        "4": 115,
        "5": 13,
        "9": 153
      },
      "local_eval": {
        "local_val_acc": 0.48412698412698413,
        "loss_after": 1.4963422389257521,
        "loss_before": 3.9004462635706343
      },
      "num_samples": 506,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 53.80739405515744
      }
    },
    {
      "cid": 4,
      "class_coverage_ratio": 0.35,
      "data_drift": {
        "corruption_type": "jpeg_compression",
        "severity": 1
      },
      "label_entropy": 1.253357109098049,
      "label_histogram": {
        "0": 2,
        "1": 48,
        "10": 1,
        "16": 3,
        "18": 35,
        "19": 13,
        "9": 1
      },
      "local_eval": {
        "local_val_acc": 0.52,
        "loss_after": 3.5089659690856934,
        "loss_before": 4.789062976837158
      },
      "num_samples": 103,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 12.251904152876932
      }
    },
    {
      "cid": 16,
      "class_coverage_ratio": 0.55,
      "data_drift": {
        "corruption_type": "clean",
        "severity": 0
      },
      "label_entropy": 1.6084735623994935,
      "label_histogram": {
        "1": 9,
        "10": 212,
        "13": 1,
        "14": 7,
        "15": 19,
        "2": 1,
        "3": 19,
        "4": 1,
        "6": 106,
        "8": 134,
        "9": 39
      },
      "local_eval": {
        "local_val_acc": 0.21897810218978103,
        "loss_after": 2.343224347942937,
        "loss_before": 5.010476731905972
      },
      "num_samples": 548,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 60.849384598522114
      }
    },
    {
      "cid": 3,
      "class_coverage_ratio": 0.5,
      "data_drift": {
        "corruption_type": "contrast",
        "severity": 1
      },
      "label_entropy": 1.0842247859989427,
      "label_histogram": {
        "1": 1,
        "10": 12,
        "11": 68,
        "12": 1,
        "14": 1,
        "16": 1,
        "2": 1,
        "4": 96,
        "5": 1,
        "8": 1
      },
      "local_eval": {
        "local_val_acc": 0.5333333333333333,
        "loss_after": 2.440294933319092,
        "loss_before": 4.5705030017428925
      },
      "num_samples": 183,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 19.24197798810251
      }
    },
    {
      "cid": 14,
      "class_coverage_ratio": 0.65,
      "data_drift": {
        "corruption_type": "contrast",
        "severity": 3
      },
      "label_entropy": 1.6982446330931011,
      "label_histogram": {
        "0": 3,
        "10": 7,
        "11": 7,
        "12": 214,
        "14": 75,
        "15": 14,
        "16": 1,
        "18": 67,
        "19": 3,
        "2": 128,
        "4": 1,
        "5": 1,
        "7": 35
      },
      "local_eval": {
        "local_val_acc": 0.4782608695652174,
        "loss_after": 1.786976959394372,
        "loss_before": 3.235879870428555
      },
      "num_samples": 556,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 60.75682553735763
      }
    },
    {
      "cid": 9,
      "class_coverage_ratio": 0.65,
      "data_drift": {
        "corruption_type": "gaussian_blur",
        "severity": 4
      },
      "label_entropy": 1.6494897617737998,
      "label_histogram": {
        "1": 17,
        "11": 110,
        "12": 11,
        "13": 6,
        "15": 1,
        "16": 1,
        "18": 1,
        "19": 146,
        "3": 2,
        "5": 64,
        "7": 195,
        "8": 7,
        "9": 3
      },
      "local_eval": {
        "local_val_acc": 0.42857142857142855,
        "loss_after": 1.954819164957319,
        "loss_before": 4.917538615635463
      },
      "num_samples": 564,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 60.813374771949334
      }
    },
    {
      "cid": 0,
      "class_coverage_ratio": 0.6,
      "data_drift": {
        "corruption_type": "clean",
        "severity": 0
      },
      "label_entropy": 1.2372041448349067,
      "label_histogram": {
        "1": 1,
        "11": 1,
        "13": 167,
        "14": 1,
        "16": 55,
        "18": 2,
        "2": 4,
        "4": 6,
        "5": 3,
        "6": 2,
        "8": 2,
        "9": 32
      },
      "local_eval": {
        "local_val_acc": 0.6956521739130435,
        "loss_after": 1.3445880844973135,
        "loss_before": 4.950259208679199
      },
      "num_samples": 276,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 29.581744174933807
      }
    },
    {
      "cid": 17,
      "class_coverage_ratio": 0.55,
      "data_drift": {
        "corruption_type": "gaussian_blur",
        "severity": 5
      },
      "label_entropy": 1.3525853257191967,
      "label_histogram": {
        "0": 324,
        "1": 1,
        "10": 1,
        "11": 78,
        "12": 1,
        "14": 8,
        "15": 56,
        "16": 12,
        "17": 1,
        "2": 155,
        "7": 3
      },
      "local_eval": {
        "local_val_acc": 0.6729559748427673,
        "loss_after": 1.3688943161154694,
        "loss_before": 4.892592490094263
      },
      "num_samples": 640,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 67.78043476055176
      }
    },
    {
      "cid": 19,
      "class_coverage_ratio": 0.6,
      "data_drift": {
        "corruption_type": "jpeg_compression",
        "severity": 3
      },
      "label_entropy": 1.6848773817720633,
      "label_histogram": {
        "0": 7,
        "1": 51,
        "10": 1,
        "12": 1,
        "14": 2,
        "15": 144,
        "3": 2,
        "4": 5,
        "5": 34,
        "7": 28,
        "8": 91,
        "9": 6
      },
      "local_eval": {
        "local_val_acc": 0.40860215053763443,
        "loss_after": 2.054769105808709,
        "loss_before": 4.401317206762171
      },
      "num_samples": 372,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 40.0174879540522
      }
    },
    {
      "cid": 7,
      "class_coverage_ratio": 0.6,
      "data_drift": {
        "corruption_type": "shot_noise",
        "severity": 4
      },
      "label_entropy": 1.4361544730284694,
      "label_histogram": {
        "0": 1,
        "10": 24,
        "11": 1,
        "13": 1,
        "15": 1,
        "16": 63,
        "17": 3,
        "2": 7,
        "3": 315,
        "4": 1,
        "6": 174,
        "9": 98
      },
      "local_eval": {
        "local_val_acc": 0.42441860465116277,
        "loss_after": 1.840960327969041,
        "loss_before": 4.9254561690397045
      },
      "num_samples": 689,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 74.69545643468706
      }
    },
    {
      "cid": 5,
      "class_coverage_ratio": 0.3,
      "data_drift": {
        "corruption_type": "brightness",
        "severity": 2
      },
      "label_entropy": 1.172803582731704,
      "label_histogram": {
        "11": 53,
        "15": 1,
        "2": 1,
        "4": 24,
        "5": 13,
        "8": 4
      },
      "local_eval": {
        "local_val_acc": 0.041666666666666664,
        "loss_after": 3.9561798572540283,
        "loss_before": 4.653658866882324
      },
      "num_samples": 96,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 8.819597553212393
      }
    },
    {
      "cid": 18,
      "class_coverage_ratio": 0.45,
      "data_drift": {
        "corruption_type": "brightness",
        "severity": 1
      },
      "label_entropy": 1.547003479217219,
      "label_histogram": {
        "1": 3,
        "12": 2,
        "16": 4,
        "18": 30,
        "3": 45,
        "4": 3,
        "7": 72,
        "8": 14,
        "9": 2
      },
      "local_eval": {
        "local_val_acc": 0.4186046511627907,
        "loss_after": 3.29504094012948,
        "loss_before": 5.054455158322356
      },
      "num_samples": 175,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 19.273126190630563
      }
    },
    {
      "cid": 10,
      "class_coverage_ratio": 0.55,
      "data_drift": {
        "corruption_type": "brightness",
        "severity": 1
      },
      "label_entropy": 1.7228936791234393,
      "label_histogram": {
        "0": 2,
        "10": 107,
        "11": 15,
        "12": 1,
        "13": 2,
        "15": 38,
        "16": 7,
        "2": 27,
        "4": 9,
        "8": 64,
        "9": 4
      },
      "local_eval": {
        "local_val_acc": 0.34782608695652173,
        "loss_after": 2.226917596830838,
        "loss_before": 4.8581610486127325
      },
      "num_samples": 276,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 29.708548094614763
      }
    }
  ],
  "clients_per_round": 5,
  "global_task": 1,
  "round_id": 2,
  "seen_classes": [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19
  ]
}'''
prompt_3='''You are selecting clients for federated continual learning.
A good client subset should:
1. Cover the currently seen classes as evenly as possible.
2. Include clients containing recently introduced or frontier classes so the model adapts to new concepts.
3. Include some clients containing previously seen classes to reduce forgetting.
4. Avoid selecting many clients with the same dominant labels.
5. Cover diverse corruption conditions and severity levels.
6. Prefer clients with positive local learning progress.
7. Avoid selecting too many unstable clients with extreme update norms or conflicting update directions.
You must output strict JSON only.
Output schema:
{
  "selected_clients": [0, 1, 2],
  "reason": "short explanation",
  "selection_summary": {
    "class_coverage": "short text",
    "drift_diversity": "short text",
    "stability": "short text"
  }
}

Input:
{
  "candidate_client_reports": [
    {
      "cid": 8,
      "class_coverage_ratio": 0.45,
      "data_drift": {
        "corruption_type": "contrast",
        "severity": 2
      },
      "label_entropy": 1.2445300896481535,
      "label_histogram": {
        "1": 251,
        "10": 2,
        "17": 3,
        "18": 93,
        "2": 2,
        "3": 1,
        "4": 72,
        "6": 19,
        "9": 8
      },
      "local_eval": {
        "local_val_acc": 0.6785714285714286,
        "loss_after": 1.6156166281018938,
        "loss_before": 4.518183299473354
      },
      "num_samples": 451,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 50.46880321481893
      }
    },
    {
      "cid": 4,
      "class_coverage_ratio": 0.35,
      "data_drift": {
        "corruption_type": "jpeg_compression",
        "severity": 1
      },
      "label_entropy": 1.253357109098049,
      "label_histogram": {
        "0": 2,
        "1": 48,
        "10": 1,
        "16": 3,
        "18": 35,
        "19": 13,
        "9": 1
      },
      "local_eval": {
        "local_val_acc": 0.32,
        "loss_after": 2.8937315940856934,
        "loss_before": 4.514346599578857
      },
      "num_samples": 103,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 12.187035612048925
      }
    },
    {
      "cid": 7,
      "class_coverage_ratio": 0.6,
      "data_drift": {
        "corruption_type": "shot_noise",
        "severity": 4
      },
      "label_entropy": 1.4361544730284694,
      "label_histogram": {
        "0": 1,
        "10": 24,
        "11": 1,
        "13": 1,
        "15": 1,
        "16": 63,
        "17": 3,
        "2": 7,
        "3": 315,
        "4": 1,
        "6": 174,
        "9": 98
      },
      "local_eval": {
        "local_val_acc": 0.46511627906976744,
        "loss_after": 1.8762156713840574,
        "loss_before": 4.60843020816182
      },
      "num_samples": 689,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 74.71239019262066
      }
    },
    {
      "cid": 14,
      "class_coverage_ratio": 0.65,
      "data_drift": {
        "corruption_type": "contrast",
        "severity": 3
      },
      "label_entropy": 1.6982446330931011,
      "label_histogram": {
        "0": 3,
        "10": 7,
        "11": 7,
        "12": 214,
        "14": 75,
        "15": 14,
        "16": 1,
        "18": 67,
        "19": 3,
        "2": 128,
        "4": 1,
        "5": 1,
        "7": 35
      },
      "local_eval": {
        "local_val_acc": 0.6376811594202898,
        "loss_after": 1.130709293095962,
        "loss_before": 3.490217927573384
      },
      "num_samples": 556,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 60.843933787934255
      }
    },
    {
      "cid": 5,
      "class_coverage_ratio": 0.3,
      "data_drift": {
        "corruption_type": "brightness",
        "severity": 2
      },
      "label_entropy": 1.172803582731704,
      "label_histogram": {
        "11": 53,
        "15": 1,
        "2": 1,
        "4": 24,
        "5": 13,
        "8": 4
      },
      "local_eval": {
        "local_val_acc": 0.0,
        "loss_after": 3.5691235065460205,
        "loss_before": 4.307875156402588
      },
      "num_samples": 96,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 8.751033971637323
      }
    },
    {
      "cid": 12,
      "class_coverage_ratio": 0.5,
      "data_drift": {
        "corruption_type": "jpeg_compression",
        "severity": 4
      },
      "label_entropy": 1.9295253971294104,
      "label_histogram": {
        "20": 31,
        "21": 7,
        "22": 3,
        "23": 59,
        "25": 17,
        "26": 11,
        "27": 5,
        "28": 16,
        "31": 26,
        "36": 3
      },
      "local_eval": {
        "local_val_acc": 0.0,
        "loss_after": 4.251940727233887,
        "loss_before": 5.85270205411044
      },
      "num_samples": 178,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 19.106857842428795
      }
    },
    {
      "cid": 9,
      "class_coverage_ratio": 0.65,
      "data_drift": {
        "corruption_type": "gaussian_blur",
        "severity": 4
      },
      "label_entropy": 1.6494897617737998,
      "label_histogram": {
        "1": 17,
        "11": 110,
        "12": 11,
        "13": 6,
        "15": 1,
        "16": 1,
        "18": 1,
        "19": 146,
        "3": 2,
        "5": 64,
        "7": 195,
        "8": 7,
        "9": 3
      },
      "local_eval": {
        "local_val_acc": 0.42142857142857143,
        "loss_after": 1.6747533934456962,
        "loss_before": 5.000870214189802
      },
      "num_samples": 564,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 60.755999728256654
      }
    },
    {
      "cid": 3,
      "class_coverage_ratio": 0.5,
      "data_drift": {
        "corruption_type": "contrast",
        "severity": 1
      },
      "label_entropy": 1.0842247859989427,
      "label_histogram": {
        "1": 1,
        "10": 12,
        "11": 68,
        "12": 1,
        "14": 1,
        "16": 1,
        "2": 1,
        "4": 96,
        "5": 1,
        "8": 1
      },
      "local_eval": {
        "local_val_acc": 0.5333333333333333,
        "loss_after": 2.1505888965394764,
        "loss_before": 4.148590162065294
      },
      "num_samples": 183,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 19.169165531649163
      }
    },
    {
      "cid": 17,
      "class_coverage_ratio": 0.55,
      "data_drift": {
        "corruption_type": "gaussian_blur",
        "severity": 5
      },
      "label_entropy": 1.3525853257191967,
      "label_histogram": {
        "0": 324,
        "1": 1,
        "10": 1,
        "11": 78,
        "12": 1,
        "14": 8,
        "15": 56,
        "16": 12,
        "17": 1,
        "2": 155,
        "7": 3
      },
      "local_eval": {
        "local_val_acc": 0.6477987421383647,
        "loss_after": 1.4154502187135085,
        "loss_before": 4.794416688523203
      },
      "num_samples": 640,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 67.74912042430519
      }
    },
    {
      "cid": 0,
      "class_coverage_ratio": 0.6,
      "data_drift": {
        "corruption_type": "clean",
        "severity": 0
      },
      "label_entropy": 1.2372041448349067,
      "label_histogram": {
        "1": 1,
        "11": 1,
        "13": 167,
        "14": 1,
        "16": 55,
        "18": 2,
        "2": 4,
        "4": 6,
        "5": 3,
        "6": 2,
        "8": 2,
        "9": 32
      },
      "local_eval": {
        "local_val_acc": 0.6956521739130435,
        "loss_after": 1.265566577082095,
        "loss_before": 3.7840809476548345
      },
      "num_samples": 276,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 29.551613717868438
      }
    },
    {
      "cid": 18,
      "class_coverage_ratio": 0.45,
      "data_drift": {
        "corruption_type": "brightness",
        "severity": 1
      },
      "label_entropy": 1.547003479217219,
      "label_histogram": {
        "1": 3,
        "12": 2,
        "16": 4,
        "18": 30,
        "3": 45,
        "4": 3,
        "7": 72,
        "8": 14,
        "9": 2
      },
      "local_eval": {
        "local_val_acc": 0.37209302325581395,
        "loss_after": 2.799765054569688,
        "loss_before": 4.7406266567318935
      },
      "num_samples": 175,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 19.1808771721207
      }
    },
    {
      "cid": 10,
      "class_coverage_ratio": 0.55,
      "data_drift": {
        "corruption_type": "brightness",
        "severity": 1
      },
      "label_entropy": 1.7228936791234393,
      "label_histogram": {
        "0": 2,
        "10": 107,
        "11": 15,
        "12": 1,
        "13": 2,
        "15": 38,
        "16": 7,
        "2": 27,
        "4": 9,
        "8": 64,
        "9": 4
      },
      "local_eval": {
        "local_val_acc": 0.3188405797101449,
        "loss_after": 1.9243111524029055,
        "loss_before": 3.8715430826380635
      },
      "num_samples": 276,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 29.590799645740983
      }
    },
    {
      "cid": 1,
      "class_coverage_ratio": 0.6,
      "data_drift": {
        "corruption_type": "pixelate",
        "severity": 3
      },
      "label_entropy": 1.5962783847217634,
      "label_histogram": {
        "0": 13,
        "10": 1,
        "11": 3,
        "14": 2,
        "15": 27,
        "16": 14,
        "17": 1,
        "18": 163,
        "19": 1,
        "4": 115,
        "5": 13,
        "9": 153
      },
      "local_eval": {
        "local_val_acc": 0.4126984126984127,
        "loss_after": 1.5368565056059096,
        "loss_before": 3.4663236784556557
      },
      "num_samples": 506,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 53.78659840189541
      }
    },
    {
      "cid": 2,
      "class_coverage_ratio": 0.6,
      "data_drift": {
        "corruption_type": "jpeg_compression",
        "severity": 3
      },
      "label_entropy": 1.7705474114506652,
      "label_histogram": {
        "1": 1,
        "10": 18,
        "11": 38,
        "13": 3,
        "14": 1,
        "18": 3,
        "19": 10,
        "2": 19,
        "4": 1,
        "5": 5,
        "7": 59,
        "9": 1
      },
      "local_eval": {
        "local_val_acc": 0.41025641025641024,
        "loss_after": 3.248679540096185,
        "loss_before": 4.547170235560491
      },
      "num_samples": 159,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 15.644617176373453
      }
    },
    {
      "cid": 13,
      "class_coverage_ratio": 0.35,
      "data_drift": {
        "corruption_type": "motion_blur",
        "severity": 5
      },
      "label_entropy": 1.0878670854864663,
      "label_histogram": {
        "11": 11,
        "14": 6,
        "17": 251,
        "18": 2,
        "2": 52,
        "6": 17,
        "8": 31
      },
      "local_eval": {
        "local_val_acc": 0.6956521739130435,
        "loss_after": 1.4544482749441396,
        "loss_before": 3.082212665806646
      },
      "num_samples": 370,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 39.90821800628404
      }
    },
    {
      "cid": 16,
      "class_coverage_ratio": 0.55,
      "data_drift": {
        "corruption_type": "clean",
        "severity": 0
      },
      "label_entropy": 1.6084735623994935,
      "label_histogram": {
        "1": 9,
        "10": 212,
        "13": 1,
        "14": 7,
        "15": 19,
        "2": 1,
        "3": 19,
        "4": 1,
        "6": 106,
        "8": 134,
        "9": 39
      },
      "local_eval": {
        "local_val_acc": 0.29927007299270075,
        "loss_after": 2.27684754698816,
        "loss_before": 4.2582331817515575
      },
      "num_samples": 548,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 60.773983008686905
      }
    },
    {
      "cid": 15,
      "class_coverage_ratio": 0.55,
      "data_drift": {
        "corruption_type": "pixelate",
        "severity": 2
      },
      "label_entropy": 1.5472977291225964,
      "label_histogram": {
        "0": 2,
        "1": 1,
        "10": 11,
        "11": 16,
        "13": 222,
        "14": 298,
        "15": 30,
        "17": 27,
        "19": 226,
        "4": 1,
        "8": 21
      },
      "local_eval": {
        "local_val_acc": 0.49295774647887325,
        "loss_after": 1.937614636801778,
        "loss_before": 4.735007861410508
      },
      "num_samples": 855,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 91.95385099279346
      }
    },
    {
      "cid": 11,
      "class_coverage_ratio": 0.7,
      "data_drift": {
        "corruption_type": "motion_blur",
        "severity": 3
      },
      "label_entropy": 2.131498794507011,
      "label_histogram": {
        "0": 1,
        "1": 2,
        "10": 5,
        "11": 1,
        "12": 93,
        "15": 73,
        "16": 100,
        "17": 57,
        "4": 62,
        "5": 5,
        "6": 13,
        "7": 13,
        "8": 30,
        "9": 38
      },
      "local_eval": {
        "local_val_acc": 0.4146341463414634,
        "loss_after": 2.0760484022822805,
        "loss_before": 2.9215244432774985
      },
      "num_samples": 493,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 53.770332317378326
      }
    },
    {
      "cid": 6,
      "class_coverage_ratio": 0.55,
      "data_drift": {
        "corruption_type": "pixelate",
        "severity": 1
      },
      "label_entropy": 1.1818091538526883,
      "label_histogram": {
        "0": 48,
        "1": 9,
        "11": 1,
        "13": 1,
        "14": 1,
        "15": 1,
        "17": 59,
        "3": 8,
        "4": 1,
        "5": 260,
        "9": 16
      },
      "local_eval": {
        "local_val_acc": 0.6633663366336634,
        "loss_after": 2.8494229033441827,
        "loss_before": 3.898940596250024
      },
      "num_samples": 405,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 43.39989138617616
      }
    },
    {
      "cid": 19,
      "class_coverage_ratio": 0.6,
      "data_drift": {
        "corruption_type": "jpeg_compression",
        "severity": 3
      },
      "label_entropy": 1.6848773817720633,
      "label_histogram": {
        "0": 7,
        "1": 51,
        "10": 1,
        "12": 1,
        "14": 2,
        "15": 144,
        "3": 2,
        "4": 5,
        "5": 34,
        "7": 28,
        "8": 91,
        "9": 6
      },
      "local_eval": {
        "local_val_acc": 0.41935483870967744,
        "loss_after": 2.0610744286608953,
        "loss_before": 3.0894988377889
      },
      "num_samples": 372,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 39.93423361916668
      }
    }
  ],
  "clients_per_round": 5,
  "global_task": 1,
  "round_id": 3,
  "seen_classes": [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19
  ]
}'''
prompt_4 = '''You are selecting clients for federated continual learning.
A good client subset should:
1. Cover the currently seen classes as evenly as possible.
2. Include clients containing recently introduced or frontier classes so the model adapts to new concepts.
3. Include some clients containing previously seen classes to reduce forgetting.
4. Avoid selecting many clients with the same dominant labels.
5. Cover diverse corruption conditions and severity levels.
6. Prefer clients with positive local learning progress.
7. Avoid selecting too many unstable clients with extreme update norms or conflicting update directions.
You must output strict JSON only.
Output schema:
{
  "selected_clients": [0, 1, 2],
  "reason": "short explanation",
  "selection_summary": {
    "class_coverage": "short text",
    "drift_diversity": "short text",
    "stability": "short text"
  }
}

Input:
{
  "candidate_client_reports": [
    {
      "cid": 14,
      "class_coverage_ratio": 0.65,
      "data_drift": {
        "corruption_type": "contrast",
        "severity": 3
      },
      "label_entropy": 1.6982446330931011,
      "label_histogram": {
        "0": 3,
        "10": 7,
        "11": 7,
        "12": 214,
        "14": 75,
        "15": 14,
        "16": 1,
        "18": 67,
        "19": 3,
        "2": 128,
        "4": 1,
        "5": 1,
        "7": 35
      },
      "local_eval": {
        "local_val_acc": 0.644927536231884,
        "loss_after": 1.146667638550634,
        "loss_before": 3.216025207353675
      },
      "num_samples": 556,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 60.86244639533394
      }
    },
    {
      "cid": 17,
      "class_coverage_ratio": 0.55,
      "data_drift": {
        "corruption_type": "gaussian_blur",
        "severity": 5
      },
      "label_entropy": 1.3525853257191967,
      "label_histogram": {
        "0": 324,
        "1": 1,
        "10": 1,
        "11": 78,
        "12": 1,
        "14": 8,
        "15": 56,
        "16": 12,
        "17": 1,
        "2": 155,
        "7": 3
      },
      "local_eval": {
        "local_val_acc": 0.660377358490566,
        "loss_after": 1.4979539888459932,
        "loss_before": 4.67844993663284
      },
      "num_samples": 640,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 67.76453085909175
      }
    },
    {
      "cid": 18,
      "class_coverage_ratio": 0.45,
      "data_drift": {
        "corruption_type": "brightness",
        "severity": 1
      },
      "label_entropy": 1.547003479217219,
      "label_histogram": {
        "1": 3,
        "12": 2,
        "16": 4,
        "18": 30,
        "3": 45,
        "4": 3,
        "7": 72,
        "8": 14,
        "9": 2
      },
      "local_eval": {
        "local_val_acc": 0.4418604651162791,
        "loss_after": 2.818738909654839,
        "loss_before": 4.134313727534095
      },
      "num_samples": 175,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 19.16579718607085
      }
    },
    {
      "cid": 13,
      "class_coverage_ratio": 0.35,
      "data_drift": {
        "corruption_type": "motion_blur",
        "severity": 5
      },
      "label_entropy": 1.0878670854864663,
      "label_histogram": {
        "11": 11,
        "14": 6,
        "17": 251,
        "18": 2,
        "2": 52,
        "6": 17,
        "8": 31
      },
      "local_eval": {
        "local_val_acc": 0.6521739130434783,
        "loss_after": 1.4128514009973276,
        "loss_before": 4.323035012120786
      },
      "num_samples": 370,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 39.94496013371185
      }
    },
    {
      "cid": 11,
      "class_coverage_ratio": 0.7,
      "data_drift": {
        "corruption_type": "motion_blur",
        "severity": 3
      },
      "label_entropy": 2.131498794507011,
      "label_histogram": {
        "0": 1,
        "1": 2,
        "10": 5,
        "11": 1,
        "12": 93,
        "15": 73,
        "16": 100,
        "17": 57,
        "4": 62,
        "5": 5,
        "6": 13,
        "7": 13,
        "8": 30,
        "9": 38
      },
      "local_eval": {
        "local_val_acc": 0.4065040650406504,
        "loss_after": 1.9426103336055105,
        "loss_before": 3.1158714236282723
      },
      "num_samples": 493,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 53.865955204928404
      }
    },
    {
      "cid": 0,
      "class_coverage_ratio": 0.6,
      "data_drift": {
        "corruption_type": "clean",
        "severity": 0
      },
      "label_entropy": 1.2372041448349067,
      "label_histogram": {
        "1": 1,
        "11": 1,
        "13": 167,
        "14": 1,
        "16": 55,
        "18": 2,
        "2": 4,
        "4": 6,
        "5": 3,
        "6": 2,
        "8": 2,
        "9": 32
      },
      "local_eval": {
        "local_val_acc": 0.6956521739130435,
        "loss_after": 1.1748349856639253,
        "loss_before": 4.21094584810561
      },
      "num_samples": 276,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 29.551657056029438
      }
    },
    {
      "cid": 2,
      "class_coverage_ratio": 0.6,
      "data_drift": {
        "corruption_type": "jpeg_compression",
        "severity": 3
      },
      "label_entropy": 1.7705474114506652,
      "label_histogram": {
        "1": 1,
        "10": 18,
        "11": 38,
        "13": 3,
        "14": 1,
        "18": 3,
        "19": 10,
        "2": 19,
        "4": 1,
        "5": 5,
        "7": 59,
        "9": 1
      },
      "local_eval": {
        "local_val_acc": 0.4358974358974359,
        "loss_after": 2.4062639077504477,
        "loss_before": 3.456096710302891
      },
      "num_samples": 159,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 15.632808756148556
      }
    },
    {
      "cid": 7,
      "class_coverage_ratio": 0.6,
      "data_drift": {
        "corruption_type": "shot_noise",
        "severity": 4
      },
      "label_entropy": 1.4361544730284694,
      "label_histogram": {
        "0": 1,
        "10": 24,
        "11": 1,
        "13": 1,
        "15": 1,
        "16": 63,
        "17": 3,
        "2": 7,
        "3": 315,
        "4": 1,
        "6": 174,
        "9": 98
      },
      "local_eval": {
        "local_val_acc": 0.4418604651162791,
        "loss_after": 1.6790049491926682,
        "loss_before": 4.8713053104489346
      },
      "num_samples": 689,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 74.66182239948229
      }
    },
    {
      "cid": 4,
      "class_coverage_ratio": 0.35,
      "data_drift": {
        "corruption_type": "jpeg_compression",
        "severity": 1
      },
      "label_entropy": 1.253357109098049,
      "label_histogram": {
        "0": 2,
        "1": 48,
        "10": 1,
        "16": 3,
        "18": 35,
        "19": 13,
        "9": 1
      },
      "local_eval": {
        "local_val_acc": 0.56,
        "loss_after": 1.700258493423462,
        "loss_before": 2.5544016361236572
      },
      "num_samples": 103,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 12.177704253168057
      }
    },
    {
      "cid": 15,
      "class_coverage_ratio": 0.55,
      "data_drift": {
        "corruption_type": "pixelate",
        "severity": 2
      },
      "label_entropy": 1.5472977291225964,
      "label_histogram": {
        "0": 2,
        "1": 1,
        "10": 11,
        "11": 16,
        "13": 222,
        "14": 298,
        "15": 30,
        "17": 27,
        "19": 226,
        "4": 1,
        "8": 21
      },
      "local_eval": {
        "local_val_acc": 0.539906103286385,
        "loss_after": 1.6681984064164856,
        "loss_before": 4.574126342092881
      },
      "num_samples": 855,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 91.93814420657294
      }
    },
    {
      "cid": 12,
      "class_coverage_ratio": 0.5,
      "data_drift": {
        "corruption_type": "jpeg_compression",
        "severity": 4
      },
      "label_entropy": 1.9295253971294104,
      "label_histogram": {
        "20": 31,
        "21": 7,
        "22": 3,
        "23": 59,
        "25": 17,
        "26": 11,
        "27": 5,
        "28": 16,
        "31": 26,
        "36": 3
      },
      "local_eval": {
        "local_val_acc": 0.0,
        "loss_after": 4.355936613949862,
        "loss_before": 6.048503182151101
      },
      "num_samples": 178,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 19.098258613897666
      }
    },
    {
      "cid": 6,
      "class_coverage_ratio": 0.55,
      "data_drift": {
        "corruption_type": "pixelate",
        "severity": 1
      },
      "label_entropy": 1.1818091538526883,
      "label_histogram": {
        "0": 48,
        "1": 9,
        "11": 1,
        "13": 1,
        "14": 1,
        "15": 1,
        "17": 59,
        "3": 8,
        "4": 1,
        "5": 260,
        "9": 16
      },
      "local_eval": {
        "local_val_acc": 0.6633663366336634,
        "loss_after": 2.225603944004172,
        "loss_before": 3.4961151982297993
      },
      "num_samples": 405,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 43.39137224556095
      }
    },
    {
      "cid": 8,
      "class_coverage_ratio": 0.45,
      "data_drift": {
        "corruption_type": "contrast",
        "severity": 2
      },
      "label_entropy": 1.2445300896481535,
      "label_histogram": {
        "1": 251,
        "10": 2,
        "17": 3,
        "18": 93,
        "2": 2,
        "3": 1,
        "4": 72,
        "6": 19,
        "9": 8
      },
      "local_eval": {
        "local_val_acc": 0.6696428571428571,
        "loss_after": 1.235819629260472,
        "loss_before": 2.76175856590271
      },
      "num_samples": 451,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 50.32349043826498
      }
    },
    {
      "cid": 10,
      "class_coverage_ratio": 0.55,
      "data_drift": {
        "corruption_type": "brightness",
        "severity": 1
      },
      "label_entropy": 1.7228936791234393,
      "label_histogram": {
        "0": 2,
        "10": 107,
        "11": 15,
        "12": 1,
        "13": 2,
        "15": 38,
        "16": 7,
        "2": 27,
        "4": 9,
        "8": 64,
        "9": 4
      },
      "local_eval": {
        "local_val_acc": 0.3188405797101449,
        "loss_after": 2.03368956455286,
        "loss_before": 3.9434190418409263
      },
      "num_samples": 276,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 29.59148520303747
      }
    },
    {
      "cid": 19,
      "class_coverage_ratio": 0.6,
      "data_drift": {
        "corruption_type": "jpeg_compression",
        "severity": 3
      },
      "label_entropy": 1.6848773817720633,
      "label_histogram": {
        "0": 7,
        "1": 51,
        "10": 1,
        "12": 1,
        "14": 2,
        "15": 144,
        "3": 2,
        "4": 5,
        "5": 34,
        "7": 28,
        "8": 91,
        "9": 6
      },
      "local_eval": {
        "local_val_acc": 0.45161290322580644,
        "loss_after": 1.9691477283354728,
        "loss_before": 3.0378825459429013
      },
      "num_samples": 372,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 39.93553205875693
      }
    },
    {
      "cid": 5,
      "class_coverage_ratio": 0.3,
      "data_drift": {
        "corruption_type": "brightness",
        "severity": 2
      },
      "label_entropy": 1.172803582731704,
      "label_histogram": {
        "11": 53,
        "15": 1,
        "2": 1,
        "4": 24,
        "5": 13,
        "8": 4
      },
      "local_eval": {
        "local_val_acc": 0.375,
        "loss_after": 2.1986000537872314,
        "loss_before": 2.921842336654663
      },
      "num_samples": 96,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 8.742215123858681
      }
    },
    {
      "cid": 3,
      "class_coverage_ratio": 0.5,
      "data_drift": {
        "corruption_type": "contrast",
        "severity": 1
      },
      "label_entropy": 1.0842247859989427,
      "label_histogram": {
        "1": 1,
        "10": 12,
        "11": 68,
        "12": 1,
        "14": 1,
        "16": 1,
        "2": 1,
        "4": 96,
        "5": 1,
        "8": 1
      },
      "local_eval": {
        "local_val_acc": 0.5333333333333333,
        "loss_after": 1.607555898030599,
        "loss_before": 2.809583144717746
      },
      "num_samples": 183,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 19.13571810214971
      }
    },
    {
      "cid": 9,
      "class_coverage_ratio": 0.65,
      "data_drift": {
        "corruption_type": "gaussian_blur",
        "severity": 4
      },
      "label_entropy": 1.6494897617737998,
      "label_histogram": {
        "1": 17,
        "11": 110,
        "12": 11,
        "13": 6,
        "15": 1,
        "16": 1,
        "18": 1,
        "19": 146,
        "3": 2,
        "5": 64,
        "7": 195,
        "8": 7,
        "9": 3
      },
      "local_eval": {
        "local_val_acc": 0.4714285714285714,
        "loss_after": 1.6394460371562414,
        "loss_before": 3.511575099400112
      },
      "num_samples": 564,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 60.71897842970257
      }
    },
    {
      "cid": 16,
      "class_coverage_ratio": 0.55,
      "data_drift": {
        "corruption_type": "clean",
        "severity": 0
      },
      "label_entropy": 1.6084735623994935,
      "label_histogram": {
        "1": 9,
        "10": 212,
        "13": 1,
        "14": 7,
        "15": 19,
        "2": 1,
        "3": 19,
        "4": 1,
        "6": 106,
        "8": 134,
        "9": 39
      },
      "local_eval": {
        "local_val_acc": 0.22627737226277372,
        "loss_after": 2.3344109806701216,
        "loss_before": 4.302214702550512
      },
      "num_samples": 548,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 60.78689634010254
      }
    },
    {
      "cid": 1,
      "class_coverage_ratio": 0.6,
      "data_drift": {
        "corruption_type": "pixelate",
        "severity": 3
      },
      "label_entropy": 1.5962783847217634,
      "label_histogram": {
        "0": 13,
        "10": 1,
        "11": 3,
        "14": 2,
        "15": 27,
        "16": 14,
        "17": 1,
        "18": 163,
        "19": 1,
        "4": 115,
        "5": 13,
        "9": 153
      },
      "local_eval": {
        "local_val_acc": 0.48412698412698413,
        "loss_after": 1.4792027435605488,
        "loss_before": 2.331372060473003
      },
      "num_samples": 506,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 53.74632815753934
      }
    }
  ],
  "clients_per_round": 5,
  "global_task": 1,
  "round_id": 4,
  "seen_classes": [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19
  ]
}'''
prompt_5 = '''You are selecting clients for federated continual learning.
A good client subset should:
1. Cover the currently seen classes as evenly as possible.
2. Include clients containing recently introduced or frontier classes so the model adapts to new concepts.
3. Include some clients containing previously seen classes to reduce forgetting.
4. Avoid selecting many clients with the same dominant labels.
5. Cover diverse corruption conditions and severity levels.
6. Prefer clients with positive local learning progress.
7. Avoid selecting too many unstable clients with extreme update norms or conflicting update directions.
You must output strict JSON only.
Output schema:
{
  "selected_clients": [0, 1, 2],
  "reason": "short explanation",
  "selection_summary": {
    "class_coverage": "short text",
    "drift_diversity": "short text",
    "stability": "short text"
  }
}

Input:
{
  "candidate_client_reports": [
    {
      "cid": 8,
      "class_coverage_ratio": 0.45,
      "data_drift": {
        "corruption_type": "contrast",
        "severity": 2
      },
      "label_entropy": 1.2445300896481535,
      "label_histogram": {
        "1": 251,
        "10": 2,
        "17": 3,
        "18": 93,
        "2": 2,
        "3": 1,
        "4": 72,
        "6": 19,
        "9": 8
      },
      "local_eval": {
        "local_val_acc": 0.6696428571428571,
        "loss_after": 1.3456330384526933,
        "loss_before": 3.7520149435315813
      },
      "num_samples": 451,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 50.46644734636615
      }
    },
    {
      "cid": 18,
      "class_coverage_ratio": 0.45,
      "data_drift": {
        "corruption_type": "brightness",
        "severity": 1
      },
      "label_entropy": 1.547003479217219,
      "label_histogram": {
        "1": 3,
        "12": 2,
        "16": 4,
        "18": 30,
        "3": 45,
        "4": 3,
        "7": 72,
        "8": 14,
        "9": 2
      },
      "local_eval": {
        "local_val_acc": 0.4418604651162791,
        "loss_after": 2.5802557468414307,
        "loss_before": 4.323384850524192
      },
      "num_samples": 175,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 19.173330065660213
      }
    },
    {
      "cid": 17,
      "class_coverage_ratio": 0.55,
      "data_drift": {
        "corruption_type": "gaussian_blur",
        "severity": 5
      },
      "label_entropy": 1.3525853257191967,
      "label_histogram": {
        "0": 324,
        "1": 1,
        "10": 1,
        "11": 78,
        "12": 1,
        "14": 8,
        "15": 56,
        "16": 12,
        "17": 1,
        "2": 155,
        "7": 3
      },
      "local_eval": {
        "local_val_acc": 0.6855345911949685,
        "loss_after": 1.4617281945996314,
        "loss_before": 4.523196394338548
      },
      "num_samples": 640,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 67.73560603366094
      }
    },
    {
      "cid": 15,
      "class_coverage_ratio": 0.55,
      "data_drift": {
        "corruption_type": "pixelate",
        "severity": 2
      },
      "label_entropy": 1.5472977291225964,
      "label_histogram": {
        "0": 2,
        "1": 1,
        "10": 11,
        "11": 16,
        "13": 222,
        "14": 298,
        "15": 30,
        "17": 27,
        "19": 226,
        "4": 1,
        "8": 21
      },
      "local_eval": {
        "local_val_acc": 0.5727699530516432,
        "loss_after": 1.778190402357791,
        "loss_before": 4.419920225098659
      },
      "num_samples": 855,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 91.95181153203315
      }
    },
    {
      "cid": 2,
      "class_coverage_ratio": 0.6,
      "data_drift": {
        "corruption_type": "jpeg_compression",
        "severity": 3
      },
      "label_entropy": 1.7705474114506652,
      "label_histogram": {
        "1": 1,
        "10": 18,
        "11": 38,
        "13": 3,
        "14": 1,
        "18": 3,
        "19": 10,
        "2": 19,
        "4": 1,
        "5": 5,
        "7": 59,
        "9": 1
      },
      "local_eval": {
        "local_val_acc": 0.38461538461538464,
        "loss_after": 2.508674077498607,
        "loss_before": 3.7478959621527257
      },
      "num_samples": 159,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 15.643773315753661
      }
    },
    {
      "cid": 12,
      "class_coverage_ratio": 0.5,
      "data_drift": {
        "corruption_type": "jpeg_compression",
        "severity": 4
      },
      "label_entropy": 1.9295253971294104,
      "label_histogram": {
        "20": 31,
        "21": 7,
        "22": 3,
        "23": 59,
        "25": 17,
        "26": 11,
        "27": 5,
        "28": 16,
        "31": 26,
        "36": 3
      },
      "local_eval": {
        "local_val_acc": 0.0,
        "loss_after": 4.40404428135265,
        "loss_before": 6.230161363428289
      },
      "num_samples": 178,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 19.108235108558993
      }
    },
    {
      "cid": 9,
      "class_coverage_ratio": 0.65,
      "data_drift": {
        "corruption_type": "gaussian_blur",
        "severity": 4
      },
      "label_entropy": 1.6494897617737998,
      "label_histogram": {
        "1": 17,
        "11": 110,
        "12": 11,
        "13": 6,
        "15": 1,
        "16": 1,
        "18": 1,
        "19": 146,
        "3": 2,
        "5": 64,
        "7": 195,
        "8": 7,
        "9": 3
      },
      "local_eval": {
        "local_val_acc": 0.4,
        "loss_after": 1.6103398288999284,
        "loss_before": 4.074423067910331
      },
      "num_samples": 564,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 60.720685849410124
      }
    },
    {
      "cid": 3,
      "class_coverage_ratio": 0.5,
      "data_drift": {
        "corruption_type": "contrast",
        "severity": 1
      },
      "label_entropy": 1.0842247859989427,
      "label_histogram": {
        "1": 1,
        "10": 12,
        "11": 68,
        "12": 1,
        "14": 1,
        "16": 1,
        "2": 1,
        "4": 96,
        "5": 1,
        "8": 1
      },
      "local_eval": {
        "local_val_acc": 0.5777777777777777,
        "loss_after": 1.4443713002734715,
        "loss_before": 3.4964918348524305
      },
      "num_samples": 183,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 19.165796406471493
      }
    },
    {
      "cid": 1,
      "class_coverage_ratio": 0.6,
      "data_drift": {
        "corruption_type": "pixelate",
        "severity": 3
      },
      "label_entropy": 1.5962783847217634,
      "label_histogram": {
        "0": 13,
        "10": 1,
        "11": 3,
        "14": 2,
        "15": 27,
        "16": 14,
        "17": 1,
        "18": 163,
        "19": 1,
        "4": 115,
        "5": 13,
        "9": 153
      },
      "local_eval": {
        "local_val_acc": 0.48412698412698413,
        "loss_after": 1.4810185035069783,
        "loss_before": 3.371079823327443
      },
      "num_samples": 506,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 53.774271276817935
      }
    },
    {
      "cid": 6,
      "class_coverage_ratio": 0.55,
      "data_drift": {
        "corruption_type": "pixelate",
        "severity": 1
      },
      "label_entropy": 1.1818091538526883,
      "label_histogram": {
        "0": 48,
        "1": 9,
        "11": 1,
        "13": 1,
        "14": 1,
        "15": 1,
        "17": 59,
        "3": 8,
        "4": 1,
        "5": 260,
        "9": 16
      },
      "local_eval": {
        "local_val_acc": 0.6633663366336634,
        "loss_after": 2.7711324030810065,
        "loss_before": 3.7778972186664546
      },
      "num_samples": 405,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 43.384638525362675
      }
    },
    {
      "cid": 14,
      "class_coverage_ratio": 0.65,
      "data_drift": {
        "corruption_type": "contrast",
        "severity": 3
      },
      "label_entropy": 1.6982446330931011,
      "label_histogram": {
        "0": 3,
        "10": 7,
        "11": 7,
        "12": 214,
        "14": 75,
        "15": 14,
        "16": 1,
        "18": 67,
        "19": 3,
        "2": 128,
        "4": 1,
        "5": 1,
        "7": 35
      },
      "local_eval": {
        "local_val_acc": 0.6304347826086957,
        "loss_after": 1.1742486677308013,
        "loss_before": 2.878332746201667
      },
      "num_samples": 556,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 60.78672963041209
      }
    },
    {
      "cid": 11,
      "class_coverage_ratio": 0.7,
      "data_drift": {
        "corruption_type": "motion_blur",
        "severity": 3
      },
      "label_entropy": 2.131498794507011,
      "label_histogram": {
        "0": 1,
        "1": 2,
        "10": 5,
        "11": 1,
        "12": 93,
        "15": 73,
        "16": 100,
        "17": 57,
        "4": 62,
        "5": 5,
        "6": 13,
        "7": 13,
        "8": 30,
        "9": 38
      },
      "local_eval": {
        "local_val_acc": 0.4634146341463415,
        "loss_after": 1.8683918937434996,
        "loss_before": 2.526206365445765
      },
      "num_samples": 493,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 53.737529671293274
      }
    },
    {
      "cid": 16,
      "class_coverage_ratio": 0.55,
      "data_drift": {
        "corruption_type": "clean",
        "severity": 0
      },
      "label_entropy": 1.6084735623994935,
      "label_histogram": {
        "1": 9,
        "10": 212,
        "13": 1,
        "14": 7,
        "15": 19,
        "2": 1,
        "3": 19,
        "4": 1,
        "6": 106,
        "8": 134,
        "9": 39
      },
      "local_eval": {
        "local_val_acc": 0.24817518248175183,
        "loss_after": 2.3392034882176533,
        "loss_before": 3.73352632905445
      },
      "num_samples": 548,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 60.75884372039903
      }
    },
    {
      "cid": 10,
      "class_coverage_ratio": 0.55,
      "data_drift": {
        "corruption_type": "brightness",
        "severity": 1
      },
      "label_entropy": 1.7228936791234393,
      "label_histogram": {
        "0": 2,
        "10": 107,
        "11": 15,
        "12": 1,
        "13": 2,
        "15": 38,
        "16": 7,
        "2": 27,
        "4": 9,
        "8": 64,
        "9": 4
      },
      "local_eval": {
        "local_val_acc": 0.3333333333333333,
        "loss_after": 2.1617929503537607,
        "loss_before": 3.311496427093727
      },
      "num_samples": 276,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 29.576476671895865
      }
    },
    {
      "cid": 4,
      "class_coverage_ratio": 0.35,
      "data_drift": {
        "corruption_type": "jpeg_compression",
        "severity": 1
      },
      "label_entropy": 1.253357109098049,
      "label_histogram": {
        "0": 2,
        "1": 48,
        "10": 1,
        "16": 3,
        "18": 35,
        "19": 13,
        "9": 1
      },
      "local_eval": {
        "local_val_acc": 0.56,
        "loss_after": 2.179414749145508,
        "loss_before": 3.615328311920166
      },
      "num_samples": 103,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 12.185138410241427
      }
    },
    {
      "cid": 13,
      "class_coverage_ratio": 0.35,
      "data_drift": {
        "corruption_type": "motion_blur",
        "severity": 5
      },
      "label_entropy": 1.0878670854864663,
      "label_histogram": {
        "11": 11,
        "14": 6,
        "17": 251,
        "18": 2,
        "2": 52,
        "6": 17,
        "8": 31
      },
      "local_eval": {
        "local_val_acc": 0.6739130434782609,
        "loss_after": 1.2453705901684968,
        "loss_before": 2.8067637526470683
      },
      "num_samples": 370,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 39.89636093638378
      }
    },
    {
      "cid": 0,
      "class_coverage_ratio": 0.6,
      "data_drift": {
        "corruption_type": "clean",
        "severity": 0
      },
      "label_entropy": 1.2372041448349067,
      "label_histogram": {
        "1": 1,
        "11": 1,
        "13": 167,
        "14": 1,
        "16": 55,
        "18": 2,
        "2": 4,
        "4": 6,
        "5": 3,
        "6": 2,
        "8": 2,
        "9": 32
      },
      "local_eval": {
        "local_val_acc": 0.7246376811594203,
        "loss_after": 1.1858682770659958,
        "loss_before": 3.4344686494357344
      },
      "num_samples": 276,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 29.562647961977024
      }
    },
    {
      "cid": 7,
      "class_coverage_ratio": 0.6,
      "data_drift": {
        "corruption_type": "shot_noise",
        "severity": 4
      },
      "label_entropy": 1.4361544730284694,
      "label_histogram": {
        "0": 1,
        "10": 24,
        "11": 1,
        "13": 1,
        "15": 1,
        "16": 63,
        "17": 3,
        "2": 7,
        "3": 315,
        "4": 1,
        "6": 174,
        "9": 98
      },
      "local_eval": {
        "local_val_acc": 0.4011627906976744,
        "loss_after": 1.7718223998712939,
        "loss_before": 4.778478034707003
      },
      "num_samples": 689,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 74.66155371601407
      }
    },
    {
      "cid": 5,
      "class_coverage_ratio": 0.3,
      "data_drift": {
        "corruption_type": "brightness",
        "severity": 2
      },
      "label_entropy": 1.172803582731704,
      "label_histogram": {
        "11": 53,
        "15": 1,
        "2": 1,
        "4": 24,
        "5": 13,
        "8": 4
      },
      "local_eval": {
        "local_val_acc": 0.125,
        "loss_after": 2.822153091430664,
        "loss_before": 3.6013967990875244
      },
      "num_samples": 96,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 8.750803648406604
      }
    },
    {
      "cid": 19,
      "class_coverage_ratio": 0.6,
      "data_drift": {
        "corruption_type": "jpeg_compression",
        "severity": 3
      },
      "label_entropy": 1.6848773817720633,
      "label_histogram": {
        "0": 7,
        "1": 51,
        "10": 1,
        "12": 1,
        "14": 2,
        "15": 144,
        "3": 2,
        "4": 5,
        "5": 34,
        "7": 28,
        "8": 91,
        "9": 6
      },
      "local_eval": {
        "local_val_acc": 0.41935483870967744,
        "loss_after": 1.8988887622792234,
        "loss_before": 2.6819878931968444
      },
      "num_samples": 372,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 39.92048602748027
      }
    }
  ],
  "clients_per_round": 5,
  "global_task": 1,
  "round_id": 5,
  "seen_classes": [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19
  ]
}'''
prompt_6 = '''You are selecting clients for federated continual learning.
A good client subset should:
1. Cover the currently seen classes as evenly as possible.
2. Include clients containing recently introduced or frontier classes so the model adapts to new concepts.
3. Include some clients containing previously seen classes to reduce forgetting.
4. Avoid selecting many clients with the same dominant labels.
5. Cover diverse corruption conditions and severity levels.
6. Prefer clients with positive local learning progress.
7. Avoid selecting too many unstable clients with extreme update norms or conflicting update directions.
You must output strict JSON only.
Output schema:
{
  "selected_clients": [0, 1, 2],
  "reason": "short explanation",
  "selection_summary": {
    "class_coverage": "short text",
    "drift_diversity": "short text",
    "stability": "short text"
  }
}

Input:
{
  "candidate_client_reports": [
    {
      "cid": 14,
      "class_coverage_ratio": 0.65,
      "data_drift": {
        "corruption_type": "contrast",
        "severity": 3
      },
      "label_entropy": 1.6982446330931011,
      "label_histogram": {
        "0": 3,
        "10": 7,
        "11": 7,
        "12": 214,
        "14": 75,
        "15": 14,
        "16": 1,
        "18": 67,
        "19": 3,
        "2": 128,
        "4": 1,
        "5": 1,
        "7": 35
      },
      "local_eval": {
        "local_val_acc": 0.572463768115942,
        "loss_after": 1.2046244144439697,
        "loss_before": 2.9431199198183804
      },
      "num_samples": 556,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 60.86078529002224
      }
    },
    {
      "cid": 18,
      "class_coverage_ratio": 0.45,
      "data_drift": {
        "corruption_type": "brightness",
        "severity": 1
      },
      "label_entropy": 1.547003479217219,
      "label_histogram": {
        "1": 3,
        "12": 2,
        "16": 4,
        "18": 30,
        "3": 45,
        "4": 3,
        "7": 72,
        "8": 14,
        "9": 2
      },
      "local_eval": {
        "local_val_acc": 0.3488372093023256,
        "loss_after": 2.5797475881354752,
        "loss_before": 4.040402889251709
      },
      "num_samples": 175,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 19.19961524578851
      }
    },
    {
      "cid": 5,
      "class_coverage_ratio": 0.3,
      "data_drift": {
        "corruption_type": "brightness",
        "severity": 2
      },
      "label_entropy": 1.172803582731704,
      "label_histogram": {
        "11": 53,
        "15": 1,
        "2": 1,
        "4": 24,
        "5": 13,
        "8": 4
      },
      "local_eval": {
        "local_val_acc": 0.4166666666666667,
        "loss_after": 1.9735565185546875,
        "loss_before": 2.718452215194702
      },
      "num_samples": 96,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 8.766679988792543
      }
    },
    {
      "cid": 8,
      "class_coverage_ratio": 0.45,
      "data_drift": {
        "corruption_type": "contrast",
        "severity": 2
      },
      "label_entropy": 1.2445300896481535,
      "label_histogram": {
        "1": 251,
        "10": 2,
        "17": 3,
        "18": 93,
        "2": 2,
        "3": 1,
        "4": 72,
        "6": 19,
        "9": 8
      },
      "local_eval": {
        "local_val_acc": 0.6696428571428571,
        "loss_after": 1.1681537457874842,
        "loss_before": 2.397275924682617
      },
      "num_samples": 451,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 50.35383614026946
      }
    },
    {
      "cid": 19,
      "class_coverage_ratio": 0.6,
      "data_drift": {
        "corruption_type": "jpeg_compression",
        "severity": 3
      },
      "label_entropy": 1.6848773817720633,
      "label_histogram": {
        "0": 7,
        "1": 51,
        "10": 1,
        "12": 1,
        "14": 2,
        "15": 144,
        "3": 2,
        "4": 5,
        "5": 34,
        "7": 28,
        "8": 91,
        "9": 6
      },
      "local_eval": {
        "local_val_acc": 0.45161290322580644,
        "loss_after": 1.7860592821592927,
        "loss_before": 2.781843898116901
      },
      "num_samples": 372,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 39.95441598820082
      }
    },
    {
      "cid": 0,
      "class_coverage_ratio": 0.6,
      "data_drift": {
        "corruption_type": "clean",
        "severity": 0
      },
      "label_entropy": 1.2372041448349067,
      "label_histogram": {
        "1": 1,
        "11": 1,
        "13": 167,
        "14": 1,
        "16": 55,
        "18": 2,
        "2": 4,
        "4": 6,
        "5": 3,
        "6": 2,
        "8": 2,
        "9": 32
      },
      "local_eval": {
        "local_val_acc": 0.7246376811594203,
        "loss_after": 1.2431182066599529,
        "loss_before": 3.710982647494993
      },
      "num_samples": 276,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 29.558705891648653
      }
    },
    {
      "cid": 7,
      "class_coverage_ratio": 0.6,
      "data_drift": {
        "corruption_type": "shot_noise",
        "severity": 4
      },
      "label_entropy": 1.4361544730284694,
      "label_histogram": {
        "0": 1,
        "10": 24,
        "11": 1,
        "13": 1,
        "15": 1,
        "16": 63,
        "17": 3,
        "2": 7,
        "3": 315,
        "4": 1,
        "6": 174,
        "9": 98
      },
      "local_eval": {
        "local_val_acc": 0.4476744186046512,
        "loss_after": 1.670521289803261,
        "loss_before": 4.873585889505786
      },
      "num_samples": 689,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 74.70017553425372
      }
    },
    {
      "cid": 13,
      "class_coverage_ratio": 0.35,
      "data_drift": {
        "corruption_type": "motion_blur",
        "severity": 5
      },
      "label_entropy": 1.0878670854864663,
      "label_histogram": {
        "11": 11,
        "14": 6,
        "17": 251,
        "18": 2,
        "2": 52,
        "6": 17,
        "8": 31
      },
      "local_eval": {
        "local_val_acc": 0.7065217391304348,
        "loss_after": 1.4399753653484841,
        "loss_before": 4.016051852184793
      },
      "num_samples": 370,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 39.930016760310785
      }
    },
    {
      "cid": 6,
      "class_coverage_ratio": 0.55,
      "data_drift": {
        "corruption_type": "pixelate",
        "severity": 1
      },
      "label_entropy": 1.1818091538526883,
      "label_histogram": {
        "0": 48,
        "1": 9,
        "11": 1,
        "13": 1,
        "14": 1,
        "15": 1,
        "17": 59,
        "3": 8,
        "4": 1,
        "5": 260,
        "9": 16
      },
      "local_eval": {
        "local_val_acc": 0.6534653465346535,
        "loss_after": 1.9867634253926796,
        "loss_before": 3.2132776987434615
      },
      "num_samples": 405,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 43.40479978095808
      }
    },
    {
      "cid": 12,
      "class_coverage_ratio": 0.5,
      "data_drift": {
        "corruption_type": "jpeg_compression",
        "severity": 4
      },
      "label_entropy": 1.9295253971294104,
      "label_histogram": {
        "20": 31,
        "21": 7,
        "22": 3,
        "23": 59,
        "25": 17,
        "26": 11,
        "27": 5,
        "28": 16,
        "31": 26,
        "36": 3
      },
      "local_eval": {
        "local_val_acc": 0.0,
        "loss_after": 4.745828325098211,
        "loss_before": 6.516597054221413
      },
      "num_samples": 178,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 19.09756016726042
      }
    },
    {
      "cid": 9,
      "class_coverage_ratio": 0.65,
      "data_drift": {
        "corruption_type": "gaussian_blur",
        "severity": 4
      },
      "label_entropy": 1.6494897617737998,
      "label_histogram": {
        "1": 17,
        "11": 110,
        "12": 11,
        "13": 6,
        "15": 1,
        "16": 1,
        "18": 1,
        "19": 146,
        "3": 2,
        "5": 64,
        "7": 195,
        "8": 7,
        "9": 3
      },
      "local_eval": {
        "local_val_acc": 0.42857142857142855,
        "loss_after": 1.5646311862128122,
        "loss_before": 2.9834383828299385
      },
      "num_samples": 564,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 60.70605502973811
      }
    },
    {
      "cid": 3,
      "class_coverage_ratio": 0.5,
      "data_drift": {
        "corruption_type": "contrast",
        "severity": 1
      },
      "label_entropy": 1.0842247859989427,
      "label_histogram": {
        "1": 1,
        "10": 12,
        "11": 68,
        "12": 1,
        "14": 1,
        "16": 1,
        "2": 1,
        "4": 96,
        "5": 1,
        "8": 1
      },
      "local_eval": {
        "local_val_acc": 0.5777777777777777,
        "loss_after": 1.3196139574050902,
        "loss_before": 2.359981139500936
      },
      "num_samples": 183,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 19.15518691986466
      }
    },
    {
      "cid": 15,
      "class_coverage_ratio": 0.55,
      "data_drift": {
        "corruption_type": "pixelate",
        "severity": 2
      },
      "label_entropy": 1.5472977291225964,
      "label_histogram": {
        "0": 2,
        "1": 1,
        "10": 11,
        "11": 16,
        "13": 222,
        "14": 298,
        "15": 30,
        "17": 27,
        "19": 226,
        "4": 1,
        "8": 21
      },
      "local_eval": {
        "local_val_acc": 0.5492957746478874,
        "loss_after": 1.7734464020796226,
        "loss_before": 4.37859217997448
      },
      "num_samples": 855,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 91.9512643220141
      }
    },
    {
      "cid": 11,
      "class_coverage_ratio": 0.7,
      "data_drift": {
        "corruption_type": "motion_blur",
        "severity": 3
      },
      "label_entropy": 2.131498794507011,
      "label_histogram": {
        "0": 1,
        "1": 2,
        "10": 5,
        "11": 1,
        "12": 93,
        "15": 73,
        "16": 100,
        "17": 57,
        "4": 62,
        "5": 5,
        "6": 13,
        "7": 13,
        "8": 30,
        "9": 38
      },
      "local_eval": {
        "local_val_acc": 0.34146341463414637,
        "loss_after": 2.132924913390865,
        "loss_before": 2.651200222775219
      },
      "num_samples": 493,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 53.77493469756655
      }
    },
    {
      "cid": 1,
      "class_coverage_ratio": 0.6,
      "data_drift": {
        "corruption_type": "pixelate",
        "severity": 3
      },
      "label_entropy": 1.5962783847217634,
      "label_histogram": {
        "0": 13,
        "10": 1,
        "11": 3,
        "14": 2,
        "15": 27,
        "16": 14,
        "17": 1,
        "18": 163,
        "19": 1,
        "4": 115,
        "5": 13,
        "9": 153
      },
      "local_eval": {
        "local_val_acc": 0.5396825396825397,
        "loss_after": 1.3333109049569993,
        "loss_before": 2.262033572272649
      },
      "num_samples": 506,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 53.73644975729689
      }
    },
    {
      "cid": 4,
      "class_coverage_ratio": 0.35,
      "data_drift": {
        "corruption_type": "jpeg_compression",
        "severity": 1
      },
      "label_entropy": 1.253357109098049,
      "label_histogram": {
        "0": 2,
        "1": 48,
        "10": 1,
        "16": 3,
        "18": 35,
        "19": 13,
        "9": 1
      },
      "local_eval": {
        "local_val_acc": 0.56,
        "loss_after": 1.8149631023406982,
        "loss_before": 2.2540767192840576
      },
      "num_samples": 103,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 12.190114043507853
      }
    },
    {
      "cid": 2,
      "class_coverage_ratio": 0.6,
      "data_drift": {
        "corruption_type": "jpeg_compression",
        "severity": 3
      },
      "label_entropy": 1.7705474114506652,
      "label_histogram": {
        "1": 1,
        "10": 18,
        "11": 38,
        "13": 3,
        "14": 1,
        "18": 3,
        "19": 10,
        "2": 19,
        "4": 1,
        "5": 5,
        "7": 59,
        "9": 1
      },
      "local_eval": {
        "local_val_acc": 0.5128205128205128,
        "loss_after": 2.0638754856892123,
        "loss_before": 3.00423089051858
      },
      "num_samples": 159,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 15.63075581661609
      }
    },
    {
      "cid": 16,
      "class_coverage_ratio": 0.55,
      "data_drift": {
        "corruption_type": "clean",
        "severity": 0
      },
      "label_entropy": 1.6084735623994935,
      "label_histogram": {
        "1": 9,
        "10": 212,
        "13": 1,
        "14": 7,
        "15": 19,
        "2": 1,
        "3": 19,
        "4": 1,
        "6": 106,
        "8": 134,
        "9": 39
      },
      "local_eval": {
        "local_val_acc": 0.24087591240875914,
        "loss_after": 2.363776749938074,
        "loss_before": 4.202097475093646
      },
      "num_samples": 548,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 60.831800237952535
      }
    },
    {
      "cid": 10,
      "class_coverage_ratio": 0.55,
      "data_drift": {
        "corruption_type": "brightness",
        "severity": 1
      },
      "label_entropy": 1.7228936791234393,
      "label_histogram": {
        "0": 2,
        "10": 107,
        "11": 15,
        "12": 1,
        "13": 2,
        "15": 38,
        "16": 7,
        "2": 27,
        "4": 9,
        "8": 64,
        "9": 4
      },
      "local_eval": {
        "local_val_acc": 0.34782608695652173,
        "loss_after": 1.8112907547881638,
        "loss_before": 4.044869920481807
      },
      "num_samples": 276,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 29.613718674504277
      }
    },
    {
      "cid": 17,
      "class_coverage_ratio": 0.55,
      "data_drift": {
        "corruption_type": "gaussian_blur",
        "severity": 5
      },
      "label_entropy": 1.3525853257191967,
      "label_histogram": {
        "0": 324,
        "1": 1,
        "10": 1,
        "11": 78,
        "12": 1,
        "14": 8,
        "15": 56,
        "16": 12,
        "17": 1,
        "2": 155,
        "7": 3
      },
      "local_eval": {
        "local_val_acc": 0.6855345911949685,
        "loss_after": 1.3334471572120234,
        "loss_before": 4.595931868883049
      },
      "num_samples": 640,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 67.73838345700015
      }
    }
  ],
  "clients_per_round": 5,
  "global_task": 1,
  "round_id": 6,
  "seen_classes": [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19
  ]
}'''
prompt_7 = '''You are selecting clients for federated continual learning.
A good client subset should:
1. Cover the currently seen classes as evenly as possible.
2. Include clients containing recently introduced or frontier classes so the model adapts to new concepts.
3. Include some clients containing previously seen classes to reduce forgetting.
4. Avoid selecting many clients with the same dominant labels.
5. Cover diverse corruption conditions and severity levels.
6. Prefer clients with positive local learning progress.
7. Avoid selecting too many unstable clients with extreme update norms or conflicting update directions.
You must output strict JSON only.
Output schema:
{
  "selected_clients": [0, 1, 2],
  "reason": "short explanation",
  "selection_summary": {
    "class_coverage": "short text",
    "drift_diversity": "short text",
    "stability": "short text"
  }
}

Input:
{
  "candidate_client_reports": [
    {
      "cid": 18,
      "class_coverage_ratio": 0.45,
      "data_drift": {
        "corruption_type": "brightness",
        "severity": 1
      },
      "label_entropy": 1.547003479217219,
      "label_histogram": {
        "1": 3,
        "12": 2,
        "16": 4,
        "18": 30,
        "3": 45,
        "4": 3,
        "7": 72,
        "8": 14,
        "9": 2
      },
      "local_eval": {
        "local_val_acc": 0.3953488372093023,
        "loss_after": 2.639665276505226,
        "loss_before": 4.786120891571045
      },
      "num_samples": 175,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 19.17032636461137
      }
    },
    {
      "cid": 12,
      "class_coverage_ratio": 0.5,
      "data_drift": {
        "corruption_type": "jpeg_compression",
        "severity": 4
      },
      "label_entropy": 1.9295253971294104,
      "label_histogram": {
        "20": 31,
        "21": 7,
        "22": 3,
        "23": 59,
        "25": 17,
        "26": 11,
        "27": 5,
        "28": 16,
        "31": 26,
        "36": 3
      },
      "local_eval": {
        "local_val_acc": 0.0,
        "loss_after": 4.729530594565651,
        "loss_before": 7.02450028332797
      },
      "num_samples": 178,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 19.104600072163883
      }
    },
    {
      "cid": 7,
      "class_coverage_ratio": 0.6,
      "data_drift": {
        "corruption_type": "shot_noise",
        "severity": 4
      },
      "label_entropy": 1.4361544730284694,
      "label_histogram": {
        "0": 1,
        "10": 24,
        "11": 1,
        "13": 1,
        "15": 1,
        "16": 63,
        "17": 3,
        "2": 7,
        "3": 315,
        "4": 1,
        "6": 174,
        "9": 98
      },
      "local_eval": {
        "local_val_acc": 0.43023255813953487,
        "loss_after": 1.7525661213453425,
        "loss_before": 6.425560718358949
      },
      "num_samples": 689,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 74.64254135038942
      }
    },
    {
      "cid": 8,
      "class_coverage_ratio": 0.45,
      "data_drift": {
        "corruption_type": "contrast",
        "severity": 2
      },
      "label_entropy": 1.2445300896481535,
      "label_histogram": {
        "1": 251,
        "10": 2,
        "17": 3,
        "18": 93,
        "2": 2,
        "3": 1,
        "4": 72,
        "6": 19,
        "9": 8
      },
      "local_eval": {
        "local_val_acc": 0.7053571428571429,
        "loss_after": 1.1612144027437483,
        "loss_before": 3.0715436254228865
      },
      "num_samples": 451,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 50.39702677162014
      }
    },
    {
      "cid": 9,
      "class_coverage_ratio": 0.65,
      "data_drift": {
        "corruption_type": "gaussian_blur",
        "severity": 4
      },
      "label_entropy": 1.6494897617737998,
      "label_histogram": {
        "1": 17,
        "11": 110,
        "12": 11,
        "13": 6,
        "15": 1,
        "16": 1,
        "18": 1,
        "19": 146,
        "3": 2,
        "5": 64,
        "7": 195,
        "8": 7,
        "9": 3
      },
      "local_eval": {
        "local_val_acc": 0.39285714285714285,
        "loss_after": 1.7575740269252231,
        "loss_before": 3.6272990907941547
      },
      "num_samples": 564,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 60.747954427599254
      }
    },
    {
      "cid": 13,
      "class_coverage_ratio": 0.35,
      "data_drift": {
        "corruption_type": "motion_blur",
        "severity": 5
      },
      "label_entropy": 1.0878670854864663,
      "label_histogram": {
        "11": 11,
        "14": 6,
        "17": 251,
        "18": 2,
        "2": 52,
        "6": 17,
        "8": 31
      },
      "local_eval": {
        "local_val_acc": 0.6956521739130435,
        "loss_after": 1.2750743264737336,
        "loss_before": 2.544626992681752
      },
      "num_samples": 370,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 39.908334958945126
      }
    },
    {
      "cid": 10,
      "class_coverage_ratio": 0.55,
      "data_drift": {
        "corruption_type": "brightness",
        "severity": 1
      },
      "label_entropy": 1.7228936791234393,
      "label_histogram": {
        "0": 2,
        "10": 107,
        "11": 15,
        "12": 1,
        "13": 2,
        "15": 38,
        "16": 7,
        "2": 27,
        "4": 9,
        "8": 64,
        "9": 4
      },
      "local_eval": {
        "local_val_acc": 0.3188405797101449,
        "loss_after": 2.1298571064852285,
        "loss_before": 3.298181475072667
      },
      "num_samples": 276,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 29.564591844852657
      }
    },
    {
      "cid": 15,
      "class_coverage_ratio": 0.55,
      "data_drift": {
        "corruption_type": "pixelate",
        "severity": 2
      },
      "label_entropy": 1.5472977291225964,
      "label_histogram": {
        "0": 2,
        "1": 1,
        "10": 11,
        "11": 16,
        "13": 222,
        "14": 298,
        "15": 30,
        "17": 27,
        "19": 226,
        "4": 1,
        "8": 21
      },
      "local_eval": {
        "local_val_acc": 0.48826291079812206,
        "loss_after": 1.8784062548982146,
        "loss_before": 4.146297600347671
      },
      "num_samples": 855,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 91.93710413555415
      }
    },
    {
      "cid": 5,
      "class_coverage_ratio": 0.3,
      "data_drift": {
        "corruption_type": "brightness",
        "severity": 2
      },
      "label_entropy": 1.172803582731704,
      "label_histogram": {
        "11": 53,
        "15": 1,
        "2": 1,
        "4": 24,
        "5": 13,
        "8": 4
      },
      "local_eval": {
        "local_val_acc": 0.125,
        "loss_after": 2.385899066925049,
        "loss_before": 3.2528839111328125
      },
      "num_samples": 96,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 8.74122897707191
      }
    },
    {
      "cid": 0,
      "class_coverage_ratio": 0.6,
      "data_drift": {
        "corruption_type": "clean",
        "severity": 0
      },
      "label_entropy": 1.2372041448349067,
      "label_histogram": {
        "1": 1,
        "11": 1,
        "13": 167,
        "14": 1,
        "16": 55,
        "18": 2,
        "2": 4,
        "4": 6,
        "5": 3,
        "6": 2,
        "8": 2,
        "9": 32
      },
      "local_eval": {
        "local_val_acc": 0.7101449275362319,
        "loss_after": 1.1384065894113071,
        "loss_before": 3.45929361426312
      },
      "num_samples": 276,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 29.555004674370384
      }
    },
    {
      "cid": 17,
      "class_coverage_ratio": 0.55,
      "data_drift": {
        "corruption_type": "gaussian_blur",
        "severity": 5
      },
      "label_entropy": 1.3525853257191967,
      "label_histogram": {
        "0": 324,
        "1": 1,
        "10": 1,
        "11": 78,
        "12": 1,
        "14": 8,
        "15": 56,
        "16": 12,
        "17": 1,
        "2": 155,
        "7": 3
      },
      "local_eval": {
        "local_val_acc": 0.7169811320754716,
        "loss_after": 1.3219970569670576,
        "loss_before": 4.05033656786073
      },
      "num_samples": 640,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 67.85760692917891
      }
    },
    {
      "cid": 19,
      "class_coverage_ratio": 0.6,
      "data_drift": {
        "corruption_type": "jpeg_compression",
        "severity": 3
      },
      "label_entropy": 1.6848773817720633,
      "label_histogram": {
        "0": 7,
        "1": 51,
        "10": 1,
        "12": 1,
        "14": 2,
        "15": 144,
        "3": 2,
        "4": 5,
        "5": 34,
        "7": 28,
        "8": 91,
        "9": 6
      },
      "local_eval": {
        "local_val_acc": 0.5161290322580645,
        "loss_after": 1.737127072067671,
        "loss_before": 2.4672544258897022
      },
      "num_samples": 372,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 39.89336271866968
      }
    },
    {
      "cid": 14,
      "class_coverage_ratio": 0.65,
      "data_drift": {
        "corruption_type": "contrast",
        "severity": 3
      },
      "label_entropy": 1.6982446330931011,
      "label_histogram": {
        "0": 3,
        "10": 7,
        "11": 7,
        "12": 214,
        "14": 75,
        "15": 14,
        "16": 1,
        "18": 67,
        "19": 3,
        "2": 128,
        "4": 1,
        "5": 1,
        "7": 35
      },
      "local_eval": {
        "local_val_acc": 0.5144927536231884,
        "loss_after": 1.8129886613375898,
        "loss_before": 3.92701666597007
      },
      "num_samples": 556,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 60.90840021687211
      }
    },
    {
      "cid": 11,
      "class_coverage_ratio": 0.7,
      "data_drift": {
        "corruption_type": "motion_blur",
        "severity": 3
      },
      "label_entropy": 2.131498794507011,
      "label_histogram": {
        "0": 1,
        "1": 2,
        "10": 5,
        "11": 1,
        "12": 93,
        "15": 73,
        "16": 100,
        "17": 57,
        "4": 62,
        "5": 5,
        "6": 13,
        "7": 13,
        "8": 30,
        "9": 38
      },
      "local_eval": {
        "local_val_acc": 0.35772357723577236,
        "loss_after": 2.1498886337125205,
        "loss_before": 3.179040129591779
      },
      "num_samples": 493,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 53.92349314525295
      }
    },
    {
      "cid": 2,
      "class_coverage_ratio": 0.6,
      "data_drift": {
        "corruption_type": "jpeg_compression",
        "severity": 3
      },
      "label_entropy": 1.7705474114506652,
      "label_histogram": {
        "1": 1,
        "10": 18,
        "11": 38,
        "13": 3,
        "14": 1,
        "18": 3,
        "19": 10,
        "2": 19,
        "4": 1,
        "5": 5,
        "7": 59,
        "9": 1
      },
      "local_eval": {
        "local_val_acc": 0.48717948717948717,
        "loss_after": 2.1819795278402476,
        "loss_before": 3.506432062540299
      },
      "num_samples": 159,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 15.63167510964197
      }
    },
    {
      "cid": 4,
      "class_coverage_ratio": 0.35,
      "data_drift": {
        "corruption_type": "jpeg_compression",
        "severity": 1
      },
      "label_entropy": 1.253357109098049,
      "label_histogram": {
        "0": 2,
        "1": 48,
        "10": 1,
        "16": 3,
        "18": 35,
        "19": 13,
        "9": 1
      },
      "local_eval": {
        "local_val_acc": 0.56,
        "loss_after": 1.928926944732666,
        "loss_before": 2.892169237136841
      },
      "num_samples": 103,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 12.179988890378173
      }
    },
    {
      "cid": 1,
      "class_coverage_ratio": 0.6,
      "data_drift": {
        "corruption_type": "pixelate",
        "severity": 3
      },
      "label_entropy": 1.5962783847217634,
      "label_histogram": {
        "0": 13,
        "10": 1,
        "11": 3,
        "14": 2,
        "15": 27,
        "16": 14,
        "17": 1,
        "18": 163,
        "19": 1,
        "4": 115,
        "5": 13,
        "9": 153
      },
      "local_eval": {
        "local_val_acc": 0.5238095238095238,
        "loss_after": 1.3742979954159449,
        "loss_before": 3.3240971111115956
      },
      "num_samples": 506,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 53.80197180467009
      }
    },
    {
      "cid": 6,
      "class_coverage_ratio": 0.55,
      "data_drift": {
        "corruption_type": "pixelate",
        "severity": 1
      },
      "label_entropy": 1.1818091538526883,
      "label_histogram": {
        "0": 48,
        "1": 9,
        "11": 1,
        "13": 1,
        "14": 1,
        "15": 1,
        "17": 59,
        "3": 8,
        "4": 1,
        "5": 260,
        "9": 16
      },
      "local_eval": {
        "local_val_acc": 0.6633663366336634,
        "loss_after": 1.2421254946453737,
        "loss_before": 2.361594610875196
      },
      "num_samples": 405,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 43.37085320215533
      }
    },
    {
      "cid": 16,
      "class_coverage_ratio": 0.55,
      "data_drift": {
        "corruption_type": "clean",
        "severity": 0
      },
      "label_entropy": 1.6084735623994935,
      "label_histogram": {
        "1": 9,
        "10": 212,
        "13": 1,
        "14": 7,
        "15": 19,
        "2": 1,
        "3": 19,
        "4": 1,
        "6": 106,
        "8": 134,
        "9": 39
      },
      "local_eval": {
        "local_val_acc": 0.2846715328467153,
        "loss_after": 2.1764150205319814,
        "loss_before": 3.9763349599211755
      },
      "num_samples": 548,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 60.75535028427338
      }
    },
    {
      "cid": 3,
      "class_coverage_ratio": 0.5,
      "data_drift": {
        "corruption_type": "contrast",
        "severity": 1
      },
      "label_entropy": 1.0842247859989427,
      "label_histogram": {
        "1": 1,
        "10": 12,
        "11": 68,
        "12": 1,
        "14": 1,
        "16": 1,
        "2": 1,
        "4": 96,
        "5": 1,
        "8": 1
      },
      "local_eval": {
        "local_val_acc": 0.6222222222222222,
        "loss_after": 1.3468019167582195,
        "loss_before": 3.5487775961558023
      },
      "num_samples": 183,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 19.165343806290863
      }
    }
  ],
  "clients_per_round": 5,
  "global_task": 1,
  "round_id": 7,
  "seen_classes": [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19
  ]
}'''
prompt_8 = '''You are selecting clients for federated continual learning.
A good client subset should:
1. Cover the currently seen classes as evenly as possible.
2. Include clients containing recently introduced or frontier classes so the model adapts to new concepts.
3. Include some clients containing previously seen classes to reduce forgetting.
4. Avoid selecting many clients with the same dominant labels.
5. Cover diverse corruption conditions and severity levels.
6. Prefer clients with positive local learning progress.
7. Avoid selecting too many unstable clients with extreme update norms or conflicting update directions.
You must output strict JSON only.
Output schema:
{
  "selected_clients": [0, 1, 2],
  "reason": "short explanation",
  "selection_summary": {
    "class_coverage": "short text",
    "drift_diversity": "short text",
    "stability": "short text"
  }
}

Input:
{
  "candidate_client_reports": [
    {
      "cid": 18,
      "class_coverage_ratio": 0.45,
      "data_drift": {
        "corruption_type": "brightness",
        "severity": 1
      },
      "label_entropy": 1.547003479217219,
      "label_histogram": {
        "1": 3,
        "12": 2,
        "16": 4,
        "18": 30,
        "3": 45,
        "4": 3,
        "7": 72,
        "8": 14,
        "9": 2
      },
      "local_eval": {
        "local_val_acc": 0.3023255813953488,
        "loss_after": 2.017650609792665,
        "loss_before": 4.014188844104146
      },
      "num_samples": 175,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 19.21666853806735
      }
    },
    {
      "cid": 12,
      "class_coverage_ratio": 0.5,
      "data_drift": {
        "corruption_type": "jpeg_compression",
        "severity": 4
      },
      "label_entropy": 1.9295253971294104,
      "label_histogram": {
        "20": 31,
        "21": 7,
        "22": 3,
        "23": 59,
        "25": 17,
        "26": 11,
        "27": 5,
        "28": 16,
        "31": 26,
        "36": 3
      },
      "local_eval": {
        "local_val_acc": 0.0,
        "loss_after": 4.772728356448087,
        "loss_before": 6.772813320159912
      },
      "num_samples": 178,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 19.118756752314184
      }
    },
    {
      "cid": 6,
      "class_coverage_ratio": 0.55,
      "data_drift": {
        "corruption_type": "pixelate",
        "severity": 1
      },
      "label_entropy": 1.1818091538526883,
      "label_histogram": {
        "0": 48,
        "1": 9,
        "11": 1,
        "13": 1,
        "14": 1,
        "15": 1,
        "17": 59,
        "3": 8,
        "4": 1,
        "5": 260,
        "9": 16
      },
      "local_eval": {
        "local_val_acc": 0.6633663366336634,
        "loss_after": 3.115752036028569,
        "loss_before": 4.3046180375731815
      },
      "num_samples": 405,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 43.41627034153017
      }
    },
    {
      "cid": 2,
      "class_coverage_ratio": 0.6,
      "data_drift": {
        "corruption_type": "jpeg_compression",
        "severity": 3
      },
      "label_entropy": 1.7705474114506652,
      "label_histogram": {
        "1": 1,
        "10": 18,
        "11": 38,
        "13": 3,
        "14": 1,
        "18": 3,
        "19": 10,
        "2": 19,
        "4": 1,
        "5": 5,
        "7": 59,
        "9": 1
      },
      "local_eval": {
        "local_val_acc": 0.358974358974359,
        "loss_after": 2.526147634555132,
        "loss_before": 4.144250612992507
      },
      "num_samples": 159,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 15.639705428257352
      }
    },
    {
      "cid": 13,
      "class_coverage_ratio": 0.35,
      "data_drift": {
        "corruption_type": "motion_blur",
        "severity": 5
      },
      "label_entropy": 1.0878670854864663,
      "label_histogram": {
        "11": 11,
        "14": 6,
        "17": 251,
        "18": 2,
        "2": 52,
        "6": 17,
        "8": 31
      },
      "local_eval": {
        "local_val_acc": 0.6956521739130435,
        "loss_after": 1.466244391773058,
        "loss_before": 4.520017706829568
      },
      "num_samples": 370,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 39.99782675231657
      }
    },
    {
      "cid": 14,
      "class_coverage_ratio": 0.65,
      "data_drift": {
        "corruption_type": "contrast",
        "severity": 3
      },
      "label_entropy": 1.6982446330931011,
      "label_histogram": {
        "0": 3,
        "10": 7,
        "11": 7,
        "12": 214,
        "14": 75,
        "15": 14,
        "16": 1,
        "18": 67,
        "19": 3,
        "2": 128,
        "4": 1,
        "5": 1,
        "7": 35
      },
      "local_eval": {
        "local_val_acc": 0.5434782608695652,
        "loss_after": 1.2281875411669414,
        "loss_before": 3.2275499295497285
      },
      "num_samples": 556,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 61.03710500551142
      }
    },
    {
      "cid": 11,
      "class_coverage_ratio": 0.7,
      "data_drift": {
        "corruption_type": "motion_blur",
        "severity": 3
      },
      "label_entropy": 2.131498794507011,
      "label_histogram": {
        "0": 1,
        "1": 2,
        "10": 5,
        "11": 1,
        "12": 93,
        "15": 73,
        "16": 100,
        "17": 57,
        "4": 62,
        "5": 5,
        "6": 13,
        "7": 13,
        "8": 30,
        "9": 38
      },
      "local_eval": {
        "local_val_acc": 0.37398373983739835,
        "loss_after": 2.1056042909622192,
        "loss_before": 2.557364808834665
      },
      "num_samples": 493,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 53.78923126424495
      }
    },
    {
      "cid": 1,
      "class_coverage_ratio": 0.6,
      "data_drift": {
        "corruption_type": "pixelate",
        "severity": 3
      },
      "label_entropy": 1.5962783847217634,
      "label_histogram": {
        "0": 13,
        "10": 1,
        "11": 3,
        "14": 2,
        "15": 27,
        "16": 14,
        "17": 1,
        "18": 163,
        "19": 1,
        "4": 115,
        "5": 13,
        "9": 153
      },
      "local_eval": {
        "local_val_acc": 0.48412698412698413,
        "loss_after": 1.3829330622203766,
        "loss_before": 2.190408025469099
      },
      "num_samples": 506,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 53.75132264881034
      }
    },
    {
      "cid": 3,
      "class_coverage_ratio": 0.5,
      "data_drift": {
        "corruption_type": "contrast",
        "severity": 1
      },
      "label_entropy": 1.0842247859989427,
      "label_histogram": {
        "1": 1,
        "10": 12,
        "11": 68,
        "12": 1,
        "14": 1,
        "16": 1,
        "2": 1,
        "4": 96,
        "5": 1,
        "8": 1
      },
      "local_eval": {
        "local_val_acc": 0.5555555555555556,
        "loss_after": 1.3547546413209703,
        "loss_before": 2.926128382152981
      },
      "num_samples": 183,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 19.17254697801648
      }
    },
    {
      "cid": 5,
      "class_coverage_ratio": 0.3,
      "data_drift": {
        "corruption_type": "brightness",
        "severity": 2
      },
      "label_entropy": 1.172803582731704,
      "label_histogram": {
        "11": 53,
        "15": 1,
        "2": 1,
        "4": 24,
        "5": 13,
        "8": 4
      },
      "local_eval": {
        "local_val_acc": 0.125,
        "loss_after": 2.8320834636688232,
        "loss_before": 3.398676633834839
      },
      "num_samples": 96,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 8.769932574801517
      }
    },
    {
      "cid": 17,
      "class_coverage_ratio": 0.55,
      "data_drift": {
        "corruption_type": "gaussian_blur",
        "severity": 5
      },
      "label_entropy": 1.3525853257191967,
      "label_histogram": {
        "0": 324,
        "1": 1,
        "10": 1,
        "11": 78,
        "12": 1,
        "14": 8,
        "15": 56,
        "16": 12,
        "17": 1,
        "2": 155,
        "7": 3
      },
      "local_eval": {
        "local_val_acc": 0.6666666666666666,
        "loss_after": 1.2906162660076934,
        "loss_before": 4.405861347726306
      },
      "num_samples": 640,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 67.75615378410845
      }
    },
    {
      "cid": 7,
      "class_coverage_ratio": 0.6,
      "data_drift": {
        "corruption_type": "shot_noise",
        "severity": 4
      },
      "label_entropy": 1.4361544730284694,
      "label_histogram": {
        "0": 1,
        "10": 24,
        "11": 1,
        "13": 1,
        "15": 1,
        "16": 63,
        "17": 3,
        "2": 7,
        "3": 315,
        "4": 1,
        "6": 174,
        "9": 98
      },
      "local_eval": {
        "local_val_acc": 0.46511627906976744,
        "loss_after": 1.5765846413235332,
        "loss_before": 3.1617201871650162
      },
      "num_samples": 689,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 74.5578776366659
      }
    },
    {
      "cid": 8,
      "class_coverage_ratio": 0.45,
      "data_drift": {
        "corruption_type": "contrast",
        "severity": 2
      },
      "label_entropy": 1.2445300896481535,
      "label_histogram": {
        "1": 251,
        "10": 2,
        "17": 3,
        "18": 93,
        "2": 2,
        "3": 1,
        "4": 72,
        "6": 19,
        "9": 8
      },
      "local_eval": {
        "local_val_acc": 0.6785714285714286,
        "loss_after": 1.4541310582842146,
        "loss_before": 3.689800330570766
      },
      "num_samples": 451,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 50.470175961322234
      }
    },
    {
      "cid": 10,
      "class_coverage_ratio": 0.55,
      "data_drift": {
        "corruption_type": "brightness",
        "severity": 1
      },
      "label_entropy": 1.7228936791234393,
      "label_histogram": {
        "0": 2,
        "10": 107,
        "11": 15,
        "12": 1,
        "13": 2,
        "15": 38,
        "16": 7,
        "2": 27,
        "4": 9,
        "8": 64,
        "9": 4
      },
      "local_eval": {
        "local_val_acc": 0.30434782608695654,
        "loss_after": 2.01390950921653,
        "loss_before": 3.879364359206048
      },
      "num_samples": 276,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 29.63468429715496
      }
    },
    {
      "cid": 16,
      "class_coverage_ratio": 0.55,
      "data_drift": {
        "corruption_type": "clean",
        "severity": 0
      },
      "label_entropy": 1.6084735623994935,
      "label_histogram": {
        "1": 9,
        "10": 212,
        "13": 1,
        "14": 7,
        "15": 19,
        "2": 1,
        "3": 19,
        "4": 1,
        "6": 106,
        "8": 134,
        "9": 39
      },
      "local_eval": {
        "local_val_acc": 0.24817518248175183,
        "loss_after": 2.430223402315683,
        "loss_before": 3.594052170314928
      },
      "num_samples": 548,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 60.82539652293912
      }
    },
    {
      "cid": 0,
      "class_coverage_ratio": 0.6,
      "data_drift": {
        "corruption_type": "clean",
        "severity": 0
      },
      "label_entropy": 1.2372041448349067,
      "label_histogram": {
        "1": 1,
        "11": 1,
        "13": 167,
        "14": 1,
        "16": 55,
        "18": 2,
        "2": 4,
        "4": 6,
        "5": 3,
        "6": 2,
        "8": 2,
        "9": 32
      },
      "local_eval": {
        "local_val_acc": 0.7101449275362319,
        "loss_after": 1.2574478336002515,
        "loss_before": 3.0264773921690127
      },
      "num_samples": 276,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 29.52759125562931
      }
    },
    {
      "cid": 9,
      "class_coverage_ratio": 0.65,
      "data_drift": {
        "corruption_type": "gaussian_blur",
        "severity": 4
      },
      "label_entropy": 1.6494897617737998,
      "label_histogram": {
        "1": 17,
        "11": 110,
        "12": 11,
        "13": 6,
        "15": 1,
        "16": 1,
        "18": 1,
        "19": 146,
        "3": 2,
        "5": 64,
        "7": 195,
        "8": 7,
        "9": 3
      },
      "local_eval": {
        "local_val_acc": 0.40714285714285714,
        "loss_after": 1.6942694289343698,
        "loss_before": 4.279326098305838
      },
      "num_samples": 564,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 60.7652168850277
      }
    },
    {
      "cid": 4,
      "class_coverage_ratio": 0.35,
      "data_drift": {
        "corruption_type": "jpeg_compression",
        "severity": 1
      },
      "label_entropy": 1.253357109098049,
      "label_histogram": {
        "0": 2,
        "1": 48,
        "10": 1,
        "16": 3,
        "18": 35,
        "19": 13,
        "9": 1
      },
      "local_eval": {
        "local_val_acc": 0.44,
        "loss_after": 1.867264747619629,
        "loss_before": 3.656132459640503
      },
      "num_samples": 103,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 12.17089916810246
      }
    },
    {
      "cid": 15,
      "class_coverage_ratio": 0.55,
      "data_drift": {
        "corruption_type": "pixelate",
        "severity": 2
      },
      "label_entropy": 1.5472977291225964,
      "label_histogram": {
        "0": 2,
        "1": 1,
        "10": 11,
        "11": 16,
        "13": 222,
        "14": 298,
        "15": 30,
        "17": 27,
        "19": 226,
        "4": 1,
        "8": 21
      },
      "local_eval": {
        "local_val_acc": 0.539906103286385,
        "loss_after": 1.698488435834786,
        "loss_before": 4.776325686996532
      },
      "num_samples": 855,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 92.00442729543502
      }
    },
    {
      "cid": 19,
      "class_coverage_ratio": 0.6,
      "data_drift": {
        "corruption_type": "jpeg_compression",
        "severity": 3
      },
      "label_entropy": 1.6848773817720633,
      "label_histogram": {
        "0": 7,
        "1": 51,
        "10": 1,
        "12": 1,
        "14": 2,
        "15": 144,
        "3": 2,
        "4": 5,
        "5": 34,
        "7": 28,
        "8": 91,
        "9": 6
      },
      "local_eval": {
        "local_val_acc": 0.3010752688172043,
        "loss_after": 2.502950632443992,
        "loss_before": 3.453509274349418
      },
      "num_samples": 372,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 40.0630403783331
      }
    }
  ],
  "clients_per_round": 5,
  "global_task": 1,
  "round_id": 8,
  "seen_classes": [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19
  ]
}'''
prompt_9 = '''You are selecting clients for federated continual learning.
A good client subset should:
1. Cover the currently seen classes as evenly as possible.
2. Include clients containing recently introduced or frontier classes so the model adapts to new concepts.
3. Include some clients containing previously seen classes to reduce forgetting.
4. Avoid selecting many clients with the same dominant labels.
5. Cover diverse corruption conditions and severity levels.
6. Prefer clients with positive local learning progress.
7. Avoid selecting too many unstable clients with extreme update norms or conflicting update directions.
You must output strict JSON only.
Output schema:
{
  "selected_clients": [0, 1, 2],
  "reason": "short explanation",
  "selection_summary": {
    "class_coverage": "short text",
    "drift_diversity": "short text",
    "stability": "short text"
  }
}

Input:
{
  "candidate_client_reports": [
    {
      "cid": 17,
      "class_coverage_ratio": 0.55,
      "data_drift": {
        "corruption_type": "gaussian_blur",
        "severity": 5
      },
      "label_entropy": 1.3525853257191967,
      "label_histogram": {
        "0": 324,
        "1": 1,
        "10": 1,
        "11": 78,
        "12": 1,
        "14": 8,
        "15": 56,
        "16": 12,
        "17": 1,
        "2": 155,
        "7": 3
      },
      "local_eval": {
        "local_val_acc": 0.710691823899371,
        "loss_after": 1.1704726065479734,
        "loss_before": 3.8747155696341076
      },
      "num_samples": 640,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 67.78738207504975
      }
    },
    {
      "cid": 13,
      "class_coverage_ratio": 0.35,
      "data_drift": {
        "corruption_type": "motion_blur",
        "severity": 5
      },
      "label_entropy": 1.0878670854864663,
      "label_histogram": {
        "11": 11,
        "14": 6,
        "17": 251,
        "18": 2,
        "2": 52,
        "6": 17,
        "8": 31
      },
      "local_eval": {
        "local_val_acc": 0.6956521739130435,
        "loss_after": 1.3683675786723262,
        "loss_before": 3.2561620422031567
      },
      "num_samples": 370,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 39.94034821159361
      }
    },
    {
      "cid": 18,
      "class_coverage_ratio": 0.45,
      "data_drift": {
        "corruption_type": "brightness",
        "severity": 1
      },
      "label_entropy": 1.547003479217219,
      "label_histogram": {
        "1": 3,
        "12": 2,
        "16": 4,
        "18": 30,
        "3": 45,
        "4": 3,
        "7": 72,
        "8": 14,
        "9": 2
      },
      "local_eval": {
        "local_val_acc": 0.37209302325581395,
        "loss_after": 2.1752262559047963,
        "loss_before": 4.139311180558315
      },
      "num_samples": 175,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 19.18126987069742
      }
    },
    {
      "cid": 6,
      "class_coverage_ratio": 0.55,
      "data_drift": {
        "corruption_type": "pixelate",
        "severity": 1
      },
      "label_entropy": 1.1818091538526883,
      "label_histogram": {
        "0": 48,
        "1": 9,
        "11": 1,
        "13": 1,
        "14": 1,
        "15": 1,
        "17": 59,
        "3": 8,
        "4": 1,
        "5": 260,
        "9": 16
      },
      "local_eval": {
        "local_val_acc": 0.6831683168316832,
        "loss_after": 1.97672434608535,
        "loss_before": 3.248575245979989
      },
      "num_samples": 405,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 43.388614069873526
      }
    },
    {
      "cid": 9,
      "class_coverage_ratio": 0.65,
      "data_drift": {
        "corruption_type": "gaussian_blur",
        "severity": 4
      },
      "label_entropy": 1.6494897617737998,
      "label_histogram": {
        "1": 17,
        "11": 110,
        "12": 11,
        "13": 6,
        "15": 1,
        "16": 1,
        "18": 1,
        "19": 146,
        "3": 2,
        "5": 64,
        "7": 195,
        "8": 7,
        "9": 3
      },
      "local_eval": {
        "local_val_acc": 0.44285714285714284,
        "loss_after": 1.66360513482775,
        "loss_before": 2.8279205799102782
      },
      "num_samples": 564,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 60.73956444133999
      }
    },
    {
      "cid": 16,
      "class_coverage_ratio": 0.55,
      "data_drift": {
        "corruption_type": "clean",
        "severity": 0
      },
      "label_entropy": 1.6084735623994935,
      "label_histogram": {
        "1": 9,
        "10": 212,
        "13": 1,
        "14": 7,
        "15": 19,
        "2": 1,
        "3": 19,
        "4": 1,
        "6": 106,
        "8": 134,
        "9": 39
      },
      "local_eval": {
        "local_val_acc": 0.27007299270072993,
        "loss_after": 2.0148704626264364,
        "loss_before": 4.133024783030043
      },
      "num_samples": 548,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 60.787555004958044
      }
    },
    {
      "cid": 15,
      "class_coverage_ratio": 0.55,
      "data_drift": {
        "corruption_type": "pixelate",
        "severity": 2
      },
      "label_entropy": 1.5472977291225964,
      "label_histogram": {
        "0": 2,
        "1": 1,
        "10": 11,
        "11": 16,
        "13": 222,
        "14": 298,
        "15": 30,
        "17": 27,
        "19": 226,
        "4": 1,
        "8": 21
      },
      "local_eval": {
        "local_val_acc": 0.3755868544600939,
        "loss_after": 2.5887453141906454,
        "loss_before": 4.195434932977381
      },
      "num_samples": 855,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 91.9336321517601
      }
    },
    {
      "cid": 19,
      "class_coverage_ratio": 0.6,
      "data_drift": {
        "corruption_type": "jpeg_compression",
        "severity": 3
      },
      "label_entropy": 1.6848773817720633,
      "label_histogram": {
        "0": 7,
        "1": 51,
        "10": 1,
        "12": 1,
        "14": 2,
        "15": 144,
        "3": 2,
        "4": 5,
        "5": 34,
        "7": 28,
        "8": 91,
        "9": 6
      },
      "local_eval": {
        "local_val_acc": 0.4946236559139785,
        "loss_after": 1.6401384581801712,
        "loss_before": 2.9491184296146518
      },
      "num_samples": 372,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 39.90996928810832
      }
    },
    {
      "cid": 10,
      "class_coverage_ratio": 0.55,
      "data_drift": {
        "corruption_type": "brightness",
        "severity": 1
      },
      "label_entropy": 1.7228936791234393,
      "label_histogram": {
        "0": 2,
        "10": 107,
        "11": 15,
        "12": 1,
        "13": 2,
        "15": 38,
        "16": 7,
        "2": 27,
        "4": 9,
        "8": 64,
        "9": 4
      },
      "local_eval": {
        "local_val_acc": 0.30434782608695654,
        "loss_after": 1.829877898312997,
        "loss_before": 4.076159359752268
      },
      "num_samples": 276,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 29.580088711312857
      }
    },
    {
      "cid": 11,
      "class_coverage_ratio": 0.7,
      "data_drift": {
        "corruption_type": "motion_blur",
        "severity": 3
      },
      "label_entropy": 2.131498794507011,
      "label_histogram": {
        "0": 1,
        "1": 2,
        "10": 5,
        "11": 1,
        "12": 93,
        "15": 73,
        "16": 100,
        "17": 57,
        "4": 62,
        "5": 5,
        "6": 13,
        "7": 13,
        "8": 30,
        "9": 38
      },
      "local_eval": {
        "local_val_acc": 0.3333333333333333,
        "loss_after": 2.2837403712233875,
        "loss_before": 3.2394334261979516
      },
      "num_samples": 493,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 53.854399667365115
      }
    },
    {
      "cid": 1,
      "class_coverage_ratio": 0.6,
      "data_drift": {
        "corruption_type": "pixelate",
        "severity": 3
      },
      "label_entropy": 1.5962783847217634,
      "label_histogram": {
        "0": 13,
        "10": 1,
        "11": 3,
        "14": 2,
        "15": 27,
        "16": 14,
        "17": 1,
        "18": 163,
        "19": 1,
        "4": 115,
        "5": 13,
        "9": 153
      },
      "local_eval": {
        "local_val_acc": 0.47619047619047616,
        "loss_after": 1.3302557600869074,
        "loss_before": 3.5746819253951783
      },
      "num_samples": 506,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 53.78989584604754
      }
    },
    {
      "cid": 4,
      "class_coverage_ratio": 0.35,
      "data_drift": {
        "corruption_type": "jpeg_compression",
        "severity": 1
      },
      "label_entropy": 1.253357109098049,
      "label_histogram": {
        "0": 2,
        "1": 48,
        "10": 1,
        "16": 3,
        "18": 35,
        "19": 13,
        "9": 1
      },
      "local_eval": {
        "local_val_acc": 0.56,
        "loss_after": 1.5467643737792969,
        "loss_before": 2.147773265838623
      },
      "num_samples": 103,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 12.209979608540637
      }
    },
    {
      "cid": 5,
      "class_coverage_ratio": 0.3,
      "data_drift": {
        "corruption_type": "brightness",
        "severity": 2
      },
      "label_entropy": 1.172803582731704,
      "label_histogram": {
        "11": 53,
        "15": 1,
        "2": 1,
        "4": 24,
        "5": 13,
        "8": 4
      },
      "local_eval": {
        "local_val_acc": 0.125,
        "loss_after": 2.3947267532348633,
        "loss_before": 3.5783755779266357
      },
      "num_samples": 96,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 8.75762775411269
      }
    },
    {
      "cid": 2,
      "class_coverage_ratio": 0.6,
      "data_drift": {
        "corruption_type": "jpeg_compression",
        "severity": 3
      },
      "label_entropy": 1.7705474114506652,
      "label_histogram": {
        "1": 1,
        "10": 18,
        "11": 38,
        "13": 3,
        "14": 1,
        "18": 3,
        "19": 10,
        "2": 19,
        "4": 1,
        "5": 5,
        "7": 59,
        "9": 1
      },
      "local_eval": {
        "local_val_acc": 0.46153846153846156,
        "loss_after": 1.8742524354885786,
        "loss_before": 3.116611957550049
      },
      "num_samples": 159,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 15.648598306030713
      }
    },
    {
      "cid": 3,
      "class_coverage_ratio": 0.5,
      "data_drift": {
        "corruption_type": "contrast",
        "severity": 1
      },
      "label_entropy": 1.0842247859989427,
      "label_histogram": {
        "1": 1,
        "10": 12,
        "11": 68,
        "12": 1,
        "14": 1,
        "16": 1,
        "2": 1,
        "4": 96,
        "5": 1,
        "8": 1
      },
      "local_eval": {
        "local_val_acc": 0.5555555555555556,
        "loss_after": 1.1490946001476712,
        "loss_before": 2.9138666046990287
      },
      "num_samples": 183,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 19.15616884969864
      }
    },
    {
      "cid": 14,
      "class_coverage_ratio": 0.65,
      "data_drift": {
        "corruption_type": "contrast",
        "severity": 3
      },
      "label_entropy": 1.6982446330931011,
      "label_histogram": {
        "0": 3,
        "10": 7,
        "11": 7,
        "12": 214,
        "14": 75,
        "15": 14,
        "16": 1,
        "18": 67,
        "19": 3,
        "2": 128,
        "4": 1,
        "5": 1,
        "7": 35
      },
      "local_eval": {
        "local_val_acc": 0.572463768115942,
        "loss_after": 1.1997422765994417,
        "loss_before": 3.193517373955768
      },
      "num_samples": 556,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 60.97643977146283
      }
    },
    {
      "cid": 12,
      "class_coverage_ratio": 0.5,
      "data_drift": {
        "corruption_type": "jpeg_compression",
        "severity": 4
      },
      "label_entropy": 1.9295253971294104,
      "label_histogram": {
        "20": 31,
        "21": 7,
        "22": 3,
        "23": 59,
        "25": 17,
        "26": 11,
        "27": 5,
        "28": 16,
        "31": 26,
        "36": 3
      },
      "local_eval": {
        "local_val_acc": 0.0,
        "loss_after": 4.735500899228183,
        "loss_before": 7.09453786503185
      },
      "num_samples": 178,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 19.112065160587502
      }
    },
    {
      "cid": 8,
      "class_coverage_ratio": 0.45,
      "data_drift": {
        "corruption_type": "contrast",
        "severity": 2
      },
      "label_entropy": 1.2445300896481535,
      "label_histogram": {
        "1": 251,
        "10": 2,
        "17": 3,
        "18": 93,
        "2": 2,
        "3": 1,
        "4": 72,
        "6": 19,
        "9": 8
      },
      "local_eval": {
        "local_val_acc": 0.6875,
        "loss_after": 1.0304676124027796,
        "loss_before": 1.983894211905343
      },
      "num_samples": 451,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 50.3513561719239
      }
    },
    {
      "cid": 7,
      "class_coverage_ratio": 0.6,
      "data_drift": {
        "corruption_type": "shot_noise",
        "severity": 4
      },
      "label_entropy": 1.4361544730284694,
      "label_histogram": {
        "0": 1,
        "10": 24,
        "11": 1,
        "13": 1,
        "15": 1,
        "16": 63,
        "17": 3,
        "2": 7,
        "3": 315,
        "4": 1,
        "6": 174,
        "9": 98
      },
      "local_eval": {
        "local_val_acc": 0.4069767441860465,
        "loss_after": 1.6792653909949369,
        "loss_before": 6.230824171110641
      },
      "num_samples": 689,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 74.64275898239326
      }
    },
    {
      "cid": 0,
      "class_coverage_ratio": 0.6,
      "data_drift": {
        "corruption_type": "clean",
        "severity": 0
      },
      "label_entropy": 1.2372041448349067,
      "label_histogram": {
        "1": 1,
        "11": 1,
        "13": 167,
        "14": 1,
        "16": 55,
        "18": 2,
        "2": 4,
        "4": 6,
        "5": 3,
        "6": 2,
        "8": 2,
        "9": 32
      },
      "local_eval": {
        "local_val_acc": 0.7246376811594203,
        "loss_after": 1.3343675378440083,
        "loss_before": 5.021147513735121
      },
      "num_samples": 276,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 29.572673790271118
      }
    }
  ],
  "clients_per_round": 5,
  "global_task": 1,
  "round_id": 9,
  "seen_classes": [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19
  ]
}'''
prompt_10 = '''You are selecting clients for federated continual learning.
A good client subset should:
1. Cover the currently seen classes as evenly as possible.
2. Include clients containing recently introduced or frontier classes so the model adapts to new concepts.
3. Include some clients containing previously seen classes to reduce forgetting.
4. Avoid selecting many clients with the same dominant labels.
5. Cover diverse corruption conditions and severity levels.
6. Prefer clients with positive local learning progress.
7. Avoid selecting too many unstable clients with extreme update norms or conflicting update directions.
You must output strict JSON only.
Output schema:
{
  "selected_clients": [0, 1, 2],
  "reason": "short explanation",
  "selection_summary": {
    "class_coverage": "short text",
    "drift_diversity": "short text",
    "stability": "short text"
  }
}

Input:
{
  "candidate_client_reports": [
    {
      "cid": 4,
      "class_coverage_ratio": 0.35,
      "data_drift": {
        "corruption_type": "jpeg_compression",
        "severity": 1
      },
      "label_entropy": 1.253357109098049,
      "label_histogram": {
        "0": 2,
        "1": 48,
        "10": 1,
        "16": 3,
        "18": 35,
        "19": 13,
        "9": 1
      },
      "local_eval": {
        "local_val_acc": 0.44,
        "loss_after": 2.0951642990112305,
        "loss_before": 3.5699503421783447
      },
      "num_samples": 103,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 12.151738246234899
      }
    },
    {
      "cid": 5,
      "class_coverage_ratio": 0.3,
      "data_drift": {
        "corruption_type": "brightness",
        "severity": 2
      },
      "label_entropy": 1.172803582731704,
      "label_histogram": {
        "11": 53,
        "15": 1,
        "2": 1,
        "4": 24,
        "5": 13,
        "8": 4
      },
      "local_eval": {
        "local_val_acc": 0.2916666666666667,
        "loss_after": 2.2613487243652344,
        "loss_before": 3.0744190216064453
      },
      "num_samples": 96,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 8.745476627300631
      }
    },
    {
      "cid": 7,
      "class_coverage_ratio": 0.6,
      "data_drift": {
        "corruption_type": "shot_noise",
        "severity": 4
      },
      "label_entropy": 1.4361544730284694,
      "label_histogram": {
        "0": 1,
        "10": 24,
        "11": 1,
        "13": 1,
        "15": 1,
        "16": 63,
        "17": 3,
        "2": 7,
        "3": 315,
        "4": 1,
        "6": 174,
        "9": 98
      },
      "local_eval": {
        "local_val_acc": 0.4069767441860465,
        "loss_after": 1.4745908809262653,
        "loss_before": 2.5220547165981557
      },
      "num_samples": 689,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 74.55109160142707
      }
    },
    {
      "cid": 0,
      "class_coverage_ratio": 0.6,
      "data_drift": {
        "corruption_type": "clean",
        "severity": 0
      },
      "label_entropy": 1.2372041448349067,
      "label_histogram": {
        "1": 1,
        "11": 1,
        "13": 167,
        "14": 1,
        "16": 55,
        "18": 2,
        "2": 4,
        "4": 6,
        "5": 3,
        "6": 2,
        "8": 2,
        "9": 32
      },
      "local_eval": {
        "local_val_acc": 0.7246376811594203,
        "loss_after": 1.1398757316064143,
        "loss_before": 2.661263286203578
      },
      "num_samples": 276,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 29.524837988283615
      }
    },
    {
      "cid": 13,
      "class_coverage_ratio": 0.35,
      "data_drift": {
        "corruption_type": "motion_blur",
        "severity": 5
      },
      "label_entropy": 1.0878670854864663,
      "label_histogram": {
        "11": 11,
        "14": 6,
        "17": 251,
        "18": 2,
        "2": 52,
        "6": 17,
        "8": 31
      },
      "local_eval": {
        "local_val_acc": 0.6195652173913043,
        "loss_after": 1.4121489680331687,
        "loss_before": 4.227498427681301
      },
      "num_samples": 370,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 39.99314759001446
      }
    },
    {
      "cid": 18,
      "class_coverage_ratio": 0.45,
      "data_drift": {
        "corruption_type": "brightness",
        "severity": 1
      },
      "label_entropy": 1.547003479217219,
      "label_histogram": {
        "1": 3,
        "12": 2,
        "16": 4,
        "18": 30,
        "3": 45,
        "4": 3,
        "7": 72,
        "8": 14,
        "9": 2
      },
      "local_eval": {
        "local_val_acc": 0.3953488372093023,
        "loss_after": 1.9129427366478498,
        "loss_before": 3.481755639231482
      },
      "num_samples": 175,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 19.180935733916737
      }
    },
    {
      "cid": 3,
      "class_coverage_ratio": 0.5,
      "data_drift": {
        "corruption_type": "contrast",
        "severity": 1
      },
      "label_entropy": 1.0842247859989427,
      "label_histogram": {
        "1": 1,
        "10": 12,
        "11": 68,
        "12": 1,
        "14": 1,
        "16": 1,
        "2": 1,
        "4": 96,
        "5": 1,
        "8": 1
      },
      "local_eval": {
        "local_val_acc": 0.5555555555555556,
        "loss_after": 1.2135288582907782,
        "loss_before": 2.796205160352919
      },
      "num_samples": 183,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 19.147588758886045
      }
    },
    {
      "cid": 10,
      "class_coverage_ratio": 0.55,
      "data_drift": {
        "corruption_type": "brightness",
        "severity": 1
      },
      "label_entropy": 1.7228936791234393,
      "label_histogram": {
        "0": 2,
        "10": 107,
        "11": 15,
        "12": 1,
        "13": 2,
        "15": 38,
        "16": 7,
        "2": 27,
        "4": 9,
        "8": 64,
        "9": 4
      },
      "local_eval": {
        "local_val_acc": 0.3188405797101449,
        "loss_after": 1.9908484963403232,
        "loss_before": 3.390783506891002
      },
      "num_samples": 276,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 29.590101590202075
      }
    },
    {
      "cid": 1,
      "class_coverage_ratio": 0.6,
      "data_drift": {
        "corruption_type": "pixelate",
        "severity": 3
      },
      "label_entropy": 1.5962783847217634,
      "label_histogram": {
        "0": 13,
        "10": 1,
        "11": 3,
        "14": 2,
        "15": 27,
        "16": 14,
        "17": 1,
        "18": 163,
        "19": 1,
        "4": 115,
        "5": 13,
        "9": 153
      },
      "local_eval": {
        "local_val_acc": 0.4603174603174603,
        "loss_after": 1.3202124001487854,
        "loss_before": 2.081593808673677
      },
      "num_samples": 506,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 53.761081192895716
      }
    },
    {
      "cid": 11,
      "class_coverage_ratio": 0.7,
      "data_drift": {
        "corruption_type": "motion_blur",
        "severity": 3
      },
      "label_entropy": 2.131498794507011,
      "label_histogram": {
        "0": 1,
        "1": 2,
        "10": 5,
        "11": 1,
        "12": 93,
        "15": 73,
        "16": 100,
        "17": 57,
        "4": 62,
        "5": 5,
        "6": 13,
        "7": 13,
        "8": 30,
        "9": 38
      },
      "local_eval": {
        "local_val_acc": 0.3821138211382114,
        "loss_after": 1.9664254799121763,
        "loss_before": 2.898667329695167
      },
      "num_samples": 493,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 53.86684082169213
      }
    },
    {
      "cid": 14,
      "class_coverage_ratio": 0.65,
      "data_drift": {
        "corruption_type": "contrast",
        "severity": 3
      },
      "label_entropy": 1.6982446330931011,
      "label_histogram": {
        "0": 3,
        "10": 7,
        "11": 7,
        "12": 214,
        "14": 75,
        "15": 14,
        "16": 1,
        "18": 67,
        "19": 3,
        "2": 128,
        "4": 1,
        "5": 1,
        "7": 35
      },
      "local_eval": {
        "local_val_acc": 0.572463768115942,
        "loss_after": 1.2531240565189417,
        "loss_before": 3.9084598016047822
      },
      "num_samples": 556,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 61.06532317050714
      }
    },
    {
      "cid": 6,
      "class_coverage_ratio": 0.55,
      "data_drift": {
        "corruption_type": "pixelate",
        "severity": 1
      },
      "label_entropy": 1.1818091538526883,
      "label_histogram": {
        "0": 48,
        "1": 9,
        "11": 1,
        "13": 1,
        "14": 1,
        "15": 1,
        "17": 59,
        "3": 8,
        "4": 1,
        "5": 260,
        "9": 16
      },
      "local_eval": {
        "local_val_acc": 0.6732673267326733,
        "loss_after": 2.4242488417294945,
        "loss_before": 3.8930399653935197
      },
      "num_samples": 405,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 43.40153899364566
      }
    },
    {
      "cid": 17,
      "class_coverage_ratio": 0.55,
      "data_drift": {
        "corruption_type": "gaussian_blur",
        "severity": 5
      },
      "label_entropy": 1.3525853257191967,
      "label_histogram": {
        "0": 324,
        "1": 1,
        "10": 1,
        "11": 78,
        "12": 1,
        "14": 8,
        "15": 56,
        "16": 12,
        "17": 1,
        "2": 155,
        "7": 3
      },
      "local_eval": {
        "local_val_acc": 0.6666666666666666,
        "loss_after": 1.29485019048055,
        "loss_before": 4.284897480370863
      },
      "num_samples": 640,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 67.87123372169357
      }
    },
    {
      "cid": 12,
      "class_coverage_ratio": 0.5,
      "data_drift": {
        "corruption_type": "jpeg_compression",
        "severity": 4
      },
      "label_entropy": 1.9295253971294104,
      "label_histogram": {
        "20": 31,
        "21": 7,
        "22": 3,
        "23": 59,
        "25": 17,
        "26": 11,
        "27": 5,
        "28": 16,
        "31": 26,
        "36": 3
      },
      "local_eval": {
        "local_val_acc": 0.0,
        "loss_after": 4.961380351673473,
        "loss_before": 7.295314095237038
      },
      "num_samples": 178,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 19.11081015026223
      }
    },
    {
      "cid": 15,
      "class_coverage_ratio": 0.55,
      "data_drift": {
        "corruption_type": "pixelate",
        "severity": 2
      },
      "label_entropy": 1.5472977291225964,
      "label_histogram": {
        "0": 2,
        "1": 1,
        "10": 11,
        "11": 16,
        "13": 222,
        "14": 298,
        "15": 30,
        "17": 27,
        "19": 226,
        "4": 1,
        "8": 21
      },
      "local_eval": {
        "local_val_acc": 0.5821596244131455,
        "loss_after": 1.5509122722025768,
        "loss_before": 4.553141155153373
      },
      "num_samples": 855,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 91.9503169019281
      }
    },
    {
      "cid": 9,
      "class_coverage_ratio": 0.65,
      "data_drift": {
        "corruption_type": "gaussian_blur",
        "severity": 4
      },
      "label_entropy": 1.6494897617737998,
      "label_histogram": {
        "1": 17,
        "11": 110,
        "12": 11,
        "13": 6,
        "15": 1,
        "16": 1,
        "18": 1,
        "19": 146,
        "3": 2,
        "5": 64,
        "7": 195,
        "8": 7,
        "9": 3
      },
      "local_eval": {
        "local_val_acc": 0.4714285714285714,
        "loss_after": 1.6216480561665125,
        "loss_before": 3.9227875641414096
      },
      "num_samples": 564,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 60.778416930539365
      }
    },
    {
      "cid": 16,
      "class_coverage_ratio": 0.55,
      "data_drift": {
        "corruption_type": "clean",
        "severity": 0
      },
      "label_entropy": 1.6084735623994935,
      "label_histogram": {
        "1": 9,
        "10": 212,
        "13": 1,
        "14": 7,
        "15": 19,
        "2": 1,
        "3": 19,
        "4": 1,
        "6": 106,
        "8": 134,
        "9": 39
      },
      "local_eval": {
        "local_val_acc": 0.29927007299270075,
        "loss_after": 1.9319130521621146,
        "loss_before": 2.9999583115542894
      },
      "num_samples": 548,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 60.72032626785997
      }
    },
    {
      "cid": 8,
      "class_coverage_ratio": 0.45,
      "data_drift": {
        "corruption_type": "contrast",
        "severity": 2
      },
      "label_entropy": 1.2445300896481535,
      "label_histogram": {
        "1": 251,
        "10": 2,
        "17": 3,
        "18": 93,
        "2": 2,
        "3": 1,
        "4": 72,
        "6": 19,
        "9": 8
      },
      "local_eval": {
        "local_val_acc": 0.6696428571428571,
        "loss_after": 1.6861968040466309,
        "loss_before": 3.7070420128958568
      },
      "num_samples": 451,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 50.4430285424473
      }
    },
    {
      "cid": 19,
      "class_coverage_ratio": 0.6,
      "data_drift": {
        "corruption_type": "jpeg_compression",
        "severity": 3
      },
      "label_entropy": 1.6848773817720633,
      "label_histogram": {
        "0": 7,
        "1": 51,
        "10": 1,
        "12": 1,
        "14": 2,
        "15": 144,
        "3": 2,
        "4": 5,
        "5": 34,
        "7": 28,
        "8": 91,
        "9": 6
      },
      "local_eval": {
        "local_val_acc": 0.4731182795698925,
        "loss_after": 1.9167070747703634,
        "loss_before": 3.254949677375055
      },
      "num_samples": 372,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 39.934899840797584
      }
    },
    {
      "cid": 2,
      "class_coverage_ratio": 0.6,
      "data_drift": {
        "corruption_type": "jpeg_compression",
        "severity": 3
      },
      "label_entropy": 1.7705474114506652,
      "label_histogram": {
        "1": 1,
        "10": 18,
        "11": 38,
        "13": 3,
        "14": 1,
        "18": 3,
        "19": 10,
        "2": 19,
        "4": 1,
        "5": 5,
        "7": 59,
        "9": 1
      },
      "local_eval": {
        "local_val_acc": 0.5128205128205128,
        "loss_after": 2.29442391028771,
        "loss_before": 3.858194962525979
      },
      "num_samples": 159,
      "update_stats": {
        "cosine_to_global_update": 0.0,
        "update_norm": 15.623498008086026
      }
    }
  ],
  "clients_per_round": 5,
  "global_task": 1,
  "round_id": 10,
  "seen_classes": [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19
  ]
}'''
