import torch
import pickle
from tqdm import tqdm
import numpy as np

from functions import match_tokenized_to_untokenized, get_all_subword_id


def get_token_matrix(args, model, tokenizer, dataset):

    mask_id = tokenizer.convert_tokens_to_ids(['[MASK]'])[0]
    model.eval()

    LAYER = int(args.layers)
    LAYER += 1  # embedding layer
    out = [[] for i in range(LAYER)]

    line_count = 0
    for line in tqdm(dataset.tokens):
        line_count += 1
        sentence = [x.form for x in line][1:]

        tokenized_text = tokenizer.tokenize(' '.join(sentence))
        tokenized_text.insert(0, '[CLS]')
        tokenized_text.append('[SEP]')
        print('---------- tokenized_text: ---------')
        print(tokenized_text)
        print('---------- tokenized_text: ---------')
        # Convert token to vocabulary indices
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

        mapping = match_tokenized_to_untokenized(tokenized_text, sentence)

        # 1. Generate mask indices
        all_layers_matrix_as_list = [[] for i in range(LAYER)]
        for i in range(0, len(tokenized_text)):
            id_for_all_i_tokens = get_all_subword_id(mapping, i)
            tmp_indexed_tokens = list(indexed_tokens)
            for tmp_id in id_for_all_i_tokens:
                if mapping[tmp_id] != -1:  # both CLS and SEP use -1 as id e.g., [-1, 0, 1, 2, ..., -1]
                    tmp_indexed_tokens[tmp_id] = mask_id
            one_batch = [list(tmp_indexed_tokens) for _ in range(0, len(tokenized_text))]
            for j in range(0, len(tokenized_text)):
                id_for_all_j_tokens = get_all_subword_id(mapping, j)
                for tmp_id in id_for_all_j_tokens:
                    if mapping[tmp_id] != -1:
                        one_batch[j][tmp_id] = mask_id

            print('one_batch: ')
            print(one_batch)

            # 2. Convert one batch to PyTorch tensors
            tokens_tensor = torch.tensor(one_batch)
            segments_tensor = torch.tensor([[0 for _ in one_sent] for one_sent in one_batch])
            if args.cuda:
                tokens_tensor = tokens_tensor.to('cuda')
                segments_tensor = segments_tensor.to('cuda')
                model.to('cuda')

            # 3. get all hidden states for one batch
            with torch.no_grad():
                model_outputs = model(tokens_tensor, segments_tensor)
                # last_layer = model_outputs[0]
                all_layers = model_outputs[-1]  # 12 layers + embedding layer

            # 4. get hidden states for word_i in one batch
            for k, layer in enumerate(all_layers):
                print('layer_num: %d' % (k+1))
                if args.cuda:
                    hidden_states_for_token_i = layer[:, i, :].cpu().numpy()
                else:
                    hidden_states_for_token_i = layer[:, i, :].numpy()
                all_layers_matrix_as_list[k].append(hidden_states_for_token_i)

        for k, one_layer_matrix in enumerate(all_layers_matrix_as_list):
            init_matrix = np.zeros((len(tokenized_text), len(tokenized_text)))
            for i, hidden_states in enumerate(one_layer_matrix):
                base_state = hidden_states[i]
                for j, state in enumerate(hidden_states):
                    if args.metric == 'dist':
                        init_matrix[i][j] = np.linalg.norm(base_state - state)
                    if args.metric == 'cos':
                        init_matrix[i][j] = np.dot(base_state, state) / (
                                    np.linalg.norm(base_state) * np.linalg.norm(state))
            out[k].append((line, tokenized_text, init_matrix))
    
    print('---------- line_count: ---------')
    print(line_count)
    print('---------- line_count: ---------')

    for k, _ in enumerate(out):
        k_output = args.output_file.format(args.model_type, args.metric, str(k))
        with open(k_output, 'wb') as fout:
            pickle.dump(out[k], fout)
            fout.close()


def get_word_matrix(args):
    deprels = []
    with open(args.matrix, 'rb') as f:
        results = pickle.load(f)

    for (line, tokenized_text, matrix_as_list) in tqdm(results):
        sentence = [x.form for x in line][1:]
        deprels.append([x.deprel for x in line])

        mapping = match_tokenized_to_untokenized(tokenized_text, sentence)

        init_matrix = matrix_as_list

        print('------------ print anslysis ------------')
        print('tokenized_text:')
        print(tokenized_text)
        print(len(tokenized_text))  # length = 40
        print('sentence:')
        print(sentence)
        print(len(sentence))  # length = 35
        print('deprels:')
        print(deprels)
        print(len(deprels[0]))  # length = 36 (add '-root-')
        print('mapping:')
        print(mapping)
        print(len(mapping))  # length = 40 corresponding to tokenized_text
        print('init_matrix:')
        print(init_matrix.shape)  # shape -> (40, 40) corresponding to tokenized_text
        print('----------------------------------\n')
        # exit()
        # merge subwords in one row
        merge_column_matrix = []
        for _, line in enumerate(init_matrix):
            new_row = []
            buf = []
            for j in range(0, len(line) - 1):
                buf.append(line[j])
                if mapping[j] != mapping[j + 1]:
                    new_row.append(buf[0])
                    buf = []
            merge_column_matrix.append(new_row)

        print('merge_column_matrix:')
        # print(merge_column_matrix)
        print(np.array(merge_column_matrix).shape)  # shape -> (40, 36)
        # exit()
        # merge subwords in multi rows
        # transpose the matrix so we can work with row instead of multiple rows
        merge_column_matrix = np.array(merge_column_matrix).transpose()
        merge_column_matrix = merge_column_matrix.tolist()
        final_matrix = []
        for _, line in enumerate(merge_column_matrix):
            new_row = []
            buf = []
            for j in range(0, len(line) - 1):
                buf.append(line[j])
                if mapping[j] != mapping[j + 1]:
                    if args.subword == 'sum':
                        new_row.append(sum(buf))
                    elif args.subword == 'avg':
                        new_row.append((sum(buf) / len(buf)))
                    elif args.subword == 'first':
                        new_row.append(buf[0])
                    buf = []
            final_matrix.append(new_row)

        # transpose to the original matrix
        final_matrix = np.array(final_matrix).transpose()

        print('final_matrix:')
        print(final_matrix)
        print(final_matrix.shape)  # shape -> (36, 36)
        # exit()

        if final_matrix.shape[0] == 0:
            print('find empty matrix:',sentence)
            continue
        assert final_matrix.shape[0] == final_matrix.shape[1]  # Attention 'assert' !!!
