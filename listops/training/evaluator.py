from numpy import random
import torch
import time
from ..data.tokenizer import COT_TOKEN, TOKEN_EOS, SEP, Tokenizer


class ListOpsTest:
    # COT_TOKEN = '='
    # TOKEN_EOS = 'END'
    # SEP=','
    def __init__(self, data, num_tests=1000, device='cpu', preencoded=True ):
        # self.model = model
        # use model device from weights
        self.device = device # next(iter(model.parameters())).device
        self.num_tests = num_tests
        self.test_data = self.test_data_choice(data['test'])
        # self.single_op_test_data = data['single_op_test'] if 'single_op_test' in data else None
        self.train_data = data['train']
        self.vocab = data['metadata']['vocab']
        print("train data:", self.train_data[:100])
        # self.config = model_config
        self.preencoded = preencoded # when True, train data is already encoded, otherwise we encode it here
        self.tokenizer = Tokenizer(self.vocab, sep=SEP)
        if not self.preencoded:
            print('encoding the train data...')
            self.train_data = self.tokenizer.encode(self.train_data, sep=SEP) # encode the train data
            print("train data encoded:", self.train_data[:100])
        self.total_len, self.max_test_eq_len = self.get_total_len(self.test_data)
        if 'test_single_ops' in data:
            for op, test_data in data['test_single_ops'].items():
                max_len, max_full_len = self.get_total_len(test_data)
                if max_len > self.max_test_eq_len:
                    self.max_test_eq_len = max_len
                    print(f"max_test_eq_len updated to {self.max_test_eq_len} for operation {op}")
                if max_full_len > self.total_len:
                    self.total_len = max_full_len
                    print(f"total_len updated to {self.total_len} for operation {op}")

        print(f"total_len: {self.total_len}, max_test_eq_len: {self.max_test_eq_len}")

        self.padding = self.get_padding(self.total_len)
        print(f"padding: {self.tokenizer.decode(self.padding)}")

        self.test_data_encoded, self.pad_lens = self.pad_test_data(self.test_data, max_len_factor=1.0)
        self.single_op_test_data = {}
        if 'test_single_ops' in data:
            for op, test_data in data['test_single_ops'].items():
                enc, pad = self.pad_test_data(test_data, max_len_factor=1.0)
                self.single_op_test_data[op] = {'test_data': test_data, 'data_encoded': enc, 'pad_lens': pad}

    def test_data_choice(self, test_data):
        """
        Choose a subset of the test data.
        """
        # randomly choose a subset of the test data
        print(len(test_data), 'test samples available.',  self.num_tests)
        indices = random.choice(len(test_data), self.num_tests, replace=False)
        return [test_data[i] for i in indices]

    def get_total_len(self, test_data):
        COT = self.tokenizer.encode(COT_TOKEN)[0]  # encode the END token
        max_len = 0
        max_full_len = 0
        for eq in test_data:
            test_sample = self.tokenizer.encode(eq['character']) 
            l = len(test_sample)
            if l > max_full_len:
                max_full_len = l
            # eq contains the full eq with solution, e.g.
            # 'median_10,(,0,8,median_10,(,1,9,8,),),=,median_10,(,0,8,8,),=,8,END'
            # we want to only give the until the first '='
            # try:
            ix = test_sample.index(COT)  # find the index of the COT token
            # except ValueError:
            #     # if COT is not found, use the full length
            #     ix = len(test_sample)
            test_sample = test_sample[:ix+1]
            l = len(test_sample)
            if l > max_len:
                max_len = l
        return max_len, max_full_len #max(self.config.block_size //2, max_len)

    def get_padding(self, l=100):
        # try a few times l to find a good padding string contianing 'e'
        END = self.tokenizer.encode(TOKEN_EOS)[0]  # encode the END token
        s = self.train_data[:l]
        for n in range(1,20):
            s = self.train_data[:n*l]
            # find the last 'e' (END) in the string
            try:
                i = s[::-1].index(END)
                p = s[:-i]  # get the padding part
                # print(len(p), self.decode(p) )
                if len(p) < l:
                    continue
                # print(f"Padding string found: {self.decode(p)} (length: {len(p)})")
                return p
            except ValueError:
                # if 'e' is not found, continue to the next iteration
                continue

        raise ValueError(f"Padding string not found in {s}")

    def pad_test_data(self,test_data,max_len_factor=1.0):
        """
        Pad the test data to the same length.
        Test data already contains encoded input (which is up to the first = sign): 
        ```
        {'character': 'max_10,(,median_10,(,0,3,1,),8,6,),=,max_10,(,1,8,6,),=,8,END',
        'encoded': [5, 1, 6, 1, 10, 13, 11, 2, 18, 16, 2, 3],
        'result': [18]}
        ```
        We will use the encoded input to pad the test data.
        """
        # test_data_chars = []
        test_data_encoded = []
        pad_lens = []
        max_len = int(self.total_len * max_len_factor)
        # test_inputs = []
        for eq in test_data:
            enc_data = eq['encoded'] 
            # eq contains the full eq with solution, e.g.
            # 'median_10,(,0,8,median_10,(,1,9,8,),),=,median_10,(,0,8,8,),=,8,END'
            # we want to only give the until the first '='
            pad_len = max_len - len(enc_data)
            if pad_len <= 0:
                pad_len = 0
                padding = []
            else:
                padding = self.padding[-pad_len:]

            pad_lens.append(pad_len)
            test_data_encoded.append(padding+enc_data) # append the encoded data with padding

        return torch.tensor(test_data_encoded, dtype=torch.long, device=self.device), pad_lens

    def generate(self, model, test_data_encoded, n=None, batch_size=64, max_new_tokens=100, verbose=False):
        # def generate(self, n=None, batch_size=64, max_new_tokens=100, verbose=False):
        """
        Generate the test data.
        """
        test_samples = test_data_encoded[:n] if n else test_data_encoded
        test_outputs = torch.zeros((len(test_samples), test_samples.shape[1]+max_new_tokens),  
                                    dtype=torch.long, device=self.device)
        print('\n\ntest_samples shape:', test_samples.shape, 'test_outputs shape:', test_outputs.shape)
        for i in range(0, len(test_samples), batch_size):
            if verbose:
                print(i, i+batch_size, len(test_samples), end='\r')
            if i+batch_size > len(test_samples):
                batch_size = len(test_samples)-i
            # print(batch_size)
            test_outputs[i:i+batch_size] = model.generate(test_samples[i:i+batch_size], max_new_tokens=max_new_tokens)
        return test_outputs

    def test_accuracy_old(self, model, num_tests=100, max_new_tokens=None, verbose=True):
        """
        !! Warning: This function is deprecated, use test_accuracy instead. It may have bugs. 
        Test the model on the test data.
        """
        t0 = time.time()
        if max_new_tokens is None:
            max_new_tokens = self.max_test_eq_len+0
        test_outputs = self.generate(model, n=num_tests, max_new_tokens=max_new_tokens)
        t1 = time.time()
        # print(f"Generated {len(test_outputs)} outputs in {t1-t0:.2f} seconds.")
        answers = []
        number_of_correct = 0
        for i, (ans, pad) in enumerate(zip(test_outputs, self.pad_lens)):
            ans = ans[pad:]
            answer = self.tokenizer.decode(ans.tolist())
            # eq=data['test'][i]
            eq = self.test_data[i]
            # print(len(answer), eq['character'], self.decode(eq['encoded']))
            answers.append(answer)
            # answer_id = answer.find('=')+1
            end_pos = answer.find(TOKEN_EOS)-1
            # we need to split by ',' to get the final answer
            if end_pos < 0:
                # print(f"!!!!!Warning: No END token found in answer {i}, using last token instead.")
                end_pos = len(answer)-1 
                # print(f"Answer: {self.tokenizer.decode(answer)}")
            # answer_id = answer[:end_pos].rfind(SEP) + 1  # find the last separator before the END token
            start_pos = answer[:end_pos].rfind(SEP)+1
            ans_final = answer[start_pos:end_pos]#.strip()
            # ans_final = answer[answer_id] if answer_id < len(answer) else 's'
            # print(answer)
            # print(ans_final, eq['result'][0], self.encode([ ans_final])[0], eq['result'][0])
            answers.append([eq['character'], answer, ans_final, ([ ans_final])] )
            if verbose:
                print(f"{i}: eq: {eq['character']} \n ans: {answer} \n ans final: {ans_final}, sol: {self.tokenizer.decode(eq['result'])}, {([ ans_final])[0]==eq['result'][0]}")
            if ([ans_final])[0] == eq['result'][0]:
                number_of_correct += 1

        accuracy = number_of_correct / len(test_outputs)
        if verbose:
            print(f"Accuracy: {accuracy:.2f}")

        t2 = time.time()
        # print(f"Tested {len(test_outputs)} samples in {t2-t1:.2f} seconds.")
        print(f"Time taken: {t2-t0:.2f}s, generation time: {t1-t0:.2f}s, decoding time: {t2-t1:.2f}s")
        return accuracy, answers

    def get_answer(self, ans, pad):
        END = self.tokenizer.encode(TOKEN_EOS)[0]
        answer = ans[pad:].tolist()  # remove padding
        try:
            answer_id = answer.index(END) -1
        except ValueError:
            # print(f"!!!!!Warning: No END token found in answer {i}, using last token instead.")
            answer_id = len(answer)-1 
            # print(f"Answer: {self.tokenizer.decode(answer)}")
        ans_final = answer[answer_id] #if answer_id < len(answer) else 's'
        return ans_final, answer_id, answer

    def check_answer(self, ans, pad, eq):
        """
        Check if the answer is correct.
        """
        ans_final, answer_id, answer = self.get_answer(ans, pad)
        answer_decoded = self.tokenizer.decode(answer[:answer_id+2])
        report = {'input': eq['character'], 
                'prediction': answer_decoded, 
                'final_answer': self.tokenizer.decode([ans_final]), 
                'expected': self.tokenizer.decode(eq['result']),
                }
        return ans_final == eq['result'][0], report

    def test_accuracy_old2(self, model, test_data_encoded=None, pad_lens=None,
                    num_tests=100, max_new_tokens=None, verbose=True, full_answer=False):
        """
        Test the model on the test data.
        """
        if test_data_encoded is None:
            test_data_encoded = self.test_data_encoded[:num_tests] if num_tests else self.test_data_encoded
        if pad_lens is None:
            pad_lens = self.pad_lens[:num_tests] if num_tests else self.pad_lens
        # END = self.tokenizer.encode(TOKEN_EOS)[0]
        if max_new_tokens is None:
            max_new_tokens = self.max_test_eq_len+0
        t0 = time.time()
        test_outputs = self.generate(model, test_data_encoded, n=num_tests, max_new_tokens=max_new_tokens)
        t1 = time.time()
        answers = []
        number_of_correct = 0
        for i, (ans, pad) in enumerate(zip(test_outputs, pad_lens)):
            eq = self.test_data[i]
            correct, report = self.check_answer(ans, pad, eq)
            if correct:
                number_of_correct += 1
            answers.append(report)
            if verbose:
                print(i,')', answers[-1])

        accuracy = number_of_correct / len(test_outputs)
        if verbose:
            print(f"Accuracy: {accuracy:.2f}")
        t2 = time.time()
        print(f"Time taken: {t2-t0:.2f}s, generation time: {t1-t0:.2f}s, decoding time: {t2-t1:.2f}s")
        return accuracy, answers

    def test_accuracy(self, model, test_data_encoded=None, test_data=None, pad_lens=None,
                    num_tests=100, max_new_tokens=None, use_pad=True,
                    verbose=True, full_answer=False):
        """
        Test the model on the test data.
        """
        if test_data is None:
            test_data = self.test_data #[:num_tests] if num_tests else self.test_data
        if test_data_encoded is None:
            test_data_encoded = self.test_data_encoded#[:num_tests] if num_tests else self.test_data_encoded
        if pad_lens is None:
            pad_lens = self.pad_lens#[:num_tests] if num_tests else self.pad_lens
        # END = self.tokenizer.encode(TOKEN_EOS)[0]
        if max_new_tokens is None:
            max_new_tokens = self.max_test_eq_len+0

        t0 = time.time()
        if use_pad:
            test_outputs = self.generate(model, test_data_encoded, n=num_tests, max_new_tokens=max_new_tokens)
        t1 = time.time()
        answers = []
        number_of_correct = 0
        # for i, (ans, pad) in enumerate(zip(test_outputs, pad_lens)):
        for i in range(len(test_data_encoded)):
            if i >= num_tests:
                break
            x = test_data_encoded[i]
            eq = test_data[i]
            pad = pad_lens[i]
            if use_pad:
                ans = test_outputs[i]
            else:
                ans = model.generate(x[pad:].unsqueeze(0), max_new_tokens=max_new_tokens)[0]
                pad = 0  # no padding when using original data
            correct, report = self.check_answer(ans, pad, eq)
            if correct:
                number_of_correct += 1
            answers.append(report)
            if verbose:
                print(i,')', answers[-1])

        accuracy = number_of_correct / (i+1) #len(test_outputs)
        if verbose:
            print(f"Accuracy: {accuracy:.2f}")
        t2 = time.time()
        print(f"Time taken: {t2-t0:.2f}s, generation time: {t1-t0:.2f}s, decoding time: {t2-t1:.2f}s")
        return accuracy, answers

    def test_acc_single_op(self, model, op, num_tests=100, max_new_tokens=None, use_pad=True, verbose=True):
        """
        Test the model on the single operation test data.
        """
        if op not in self.single_op_test_data:
            raise ValueError(f"Operation {op} not found in single_op_test_data. Available operations: {list(self.single_op_test_data.keys())}")

        test_data = self.single_op_test_data[op]['test_data']
        test_data_encoded = self.single_op_test_data[op]['data_encoded']
        pad_lens = self.single_op_test_data[op]['pad_lens']

        return self.test_accuracy(
            model,
            test_data_encoded=test_data_encoded,
            test_data=test_data,
            pad_lens=pad_lens,
            num_tests=num_tests,
            max_new_tokens=max_new_tokens,
            use_pad=use_pad,
            verbose=verbose,
        )

    def test_accuracy_no_pad(self, model, test_data_encoded=None, test_data=None, pad_lens=None,
                            num_tests=100, max_new_tokens=None, verbose=True):
        """
        Test the model on the test data. 
        Instead of using padded data, use the original data.
        """
        import warnings
        warnings.warn("test_accuracy_no_pad is deprecated, use test_accuracy instead")
        # test_outputs = self.generate(n=num_tests)
        # instead
        # n = num_tests
        # test_samples = self.test_data_encoded[:num_tests] if n else self.test_data_encoded
        if test_data is None:
            test_data = self.test_data[:num_tests] if num_tests else self.test_data
        if test_data_encoded is None:
            test_data_encoded = self.test_data_encoded[:num_tests] if num_tests else self.test_data_encoded
        if pad_lens is None:
            pad_lens = self.pad_lens[:num_tests] if num_tests else self.pad_lens
        if max_new_tokens is None:
            max_new_tokens = self.max_test_eq_len+0
        answers = []
        # debug_output = []
        number_of_correct = 0
        t0 = time.time()
        for i, (x, pad) in enumerate(zip(test_data_encoded, pad_lens)):
            if i >= num_tests:
                break
            ans = model.generate(x[pad:].unsqueeze(0), max_new_tokens=max_new_tokens)[0]
            eq = self.test_data[i]
            correct, report = self.check_answer(ans=ans, pad=0, eq=eq)
            if correct:
                number_of_correct += 1
            answers.append(report)
            if verbose:
                print(i,')', answers[-1])

        accuracy = number_of_correct / num_tests
        if verbose:
            print(f"Accuracy: {accuracy:.2f}")
        print(f"Test Time: {time.time()-t0:.2f}s")
        return accuracy, answers, #debug_output
