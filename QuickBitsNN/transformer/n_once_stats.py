import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers



#
# ==================================================================# ==================================================================
# ==================================================================# ==================================================================
class NonceDigitStats(nn.Module):
    # -----------------
    def __init__(self, device=f'cuda:0', n_digits = 4):
        super().__init__()
        self.device = device
        self.n_digits = n_digits
        self.n_onces = torch.tensor(0, device=self.device)
        self.n_equal_tokens_per_digit = torch.zeros((self.n_digits,), dtype=torch.long, device=self.device)

        self.harmonic_mean = lambda x_list: len(x_list) / sum([1./x_ for x_ in x_list])
        # Running Sum in format for Harmonic Mean (sum(1/x))
        self.n_prob = torch.tensor(0, dtype=torch.long, device=self.device)
        self.total_prob_of_labels = torch.tensor(0., dtype=torch.float32, device=self.device)
        self.known_labels_prob_per_digit = torch.zeros((self.n_digits,), dtype=torch.float32, device=self.device)
        self.total_logprob_of_gen_once = torch.tensor(0., dtype=torch.float32, device=self.device)
        self.best_run_acc = {}
        self.best_run_acc['digit_accuracy'] = torch.zeros((self.n_digits,), dtype=torch.float32, device=self.device)
        self.best_run_acc['percent_above_rand'] = torch.zeros((1,), dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def accumulate_eqtokens_per_label_digit(self, nonce_for_eq: torch.Tensor, nonce_labels: torch.Tensor) -> None:
        """ Assume that: nonce_for_eq = restructure_generated_nonces_for_eq_tokens(candidate_nonce) """
        device = nonce_for_eq.device
        self.n_onces = self.n_onces.to(device=device)
    
        self.n_equal_tokens_per_digit = self.n_equal_tokens_per_digit.to(device=device)
        gen_nonce_eq_tokens = [torch.eq(nonce_for_eq[:, n_].reshape(-1), nonce_labels[:,n_].reshape(-1)).to(dtype=torch.long) for n_ in range(nonce_for_eq.size(1))]
        eq_tokens_by_digit = torch.stack(gen_nonce_eq_tokens, dim=0)
        eq_tokens_by_digit = eq_tokens_by_digit.permute(1,0)
        self.n_equal_tokens_per_digit += eq_tokens_by_digit.sum(0)
        self.n_onces += nonce_for_eq.size(0)
        return

    # -------------------------------------------
    def logit_argmax_eqtokens_per_label_digit(self, logits: torch.Tensor, nonce_labels: torch.Tensor) -> float:
        #logit_classes = logits.argmax(1)
        logit_classes = logits.argmax(1)
        assert len(logit_classes.reshape(-1)) == len(nonce_labels.reshape(-1))
        accuracy = torch.eq(logit_classes.reshape(-1), nonce_labels.reshape(-1)).float().mean().item()
        return accuracy

    # ------------------------------------------------------------------
    def validation_epoch_end_average_eq_tokens_by_digit(self, log=True) -> dict[str, float]:
        """ self.tb_logger.add_scalars(f'accuracy_per_digit', tb_log, self.trainer.global_step) """
        avg_eq_tokens_by_digit = self.n_equal_tokens_per_digit / self.n_onces

        if log:
            accuracy_per_digit_log = {}
            _ = [accuracy_per_digit_log.update({f'/nonce[{i_}]': avg_eq_tokens_by_digit[i_]}) for i_ in
                 range(self.n_digits)]
            return accuracy_per_digit_log
        return avg_eq_tokens_by_digit

    # ---------------------------------
    def validation_epoch_end_eq_tokens(self) -> float:
        """  self.log('eq_tokens/average/total', avg_eq_tokens_by_nonce, on_step=False, on_epoch=True) """
        avg_eq_tokens = self.n_equal_tokens_per_digit.sum(0) / (self.n_digits*self.n_onces)
        return avg_eq_tokens

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def logit_known_class_probability(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        batch_ = logits.size(0)
        device = logits.device
        self.total_prob_of_labels = self.total_prob_of_labels.to(device=device)
        self.known_labels_prob_per_digit = self.known_labels_prob_per_digit.to(device=device)
        self.total_logprob_of_gen_once = self.total_logprob_of_gen_once.to(device=device)

        logit_probs = torch.softmax(logits, dim=1, dtype=torch.float32).to(device=device)
        known_class_prob = [torch.take_along_dim(logit_probs[:,:,idx_], labels[:,idx_].unsqueeze(-1), dim=-1) for idx_ in range(self.n_digits)]
        prob_by_digit = torch.cat(known_class_prob, dim=-1)

        # Total probability of the full word is Prod(Prob) = ao*a1*a2*a3 = Sum(log(Prob))
        sum_of_logprob_for_each_generated_word = ( np.power(4, -0.75) * (torch.log(prob_by_digit)).sum(1)).sum(0)
        
        # Running Sum in format for Harmonic Mean (sum(1/x))
        #sum_n_once_prob = (1.0/prob_by_digit).sum(0)     # tensor((n_digits,))
        #sum_prob_of_all_digits = (1.0/prob_by_digit.sum(1)).sum(0)   # tensor((one_item,))

        # Regular Mean Average
        sum_n_once_prob = (prob_by_digit).sum(0)     # tensor((n_digits,))
        sum_prob_of_all_digits = (prob_by_digit.sum(1)).sum(0)   # tensor((one_item,))
        
        self.n_prob += batch_
        self.total_prob_of_labels += sum_prob_of_all_digits.to(device=device)
        self.known_labels_prob_per_digit += sum_n_once_prob.to().to(device=device)
        self.total_logprob_of_gen_once += sum_of_logprob_for_each_generated_word.to(device=device)
        return

    # ---------------------------------------------------------------------
    def validation_epoch_end_known_label_prob_per_digit(self, log=True) -> dict:
        # Harmonic Mean
        #avg_prob_per_digit = self.n_prob / self.known_labels_prob_per_digit
        # Regular Mean
        avg_prob_per_digit = self.known_labels_prob_per_digit /  self.n_prob

        if log:
            prob_per_digit_log = {}
            _ = [prob_per_digit_log.update({f'/nonce[{i_}]': avg_prob_per_digit[i_]}) for i_ in
                 range(self.n_digits)]
            
            return prob_per_digit_log

        return avg_prob_per_digit

    # ------------------------------------------------------------
    def validation_epoch_end_prob_of_known_n_once(self) -> float:
        # Harmonic Mean
        #return self.n_prob / self.total_prob_of_labels
        # Regular Mean
        return  self.total_prob_of_labels / (self.n_prob * self.n_digits)
        
        
    # ------------------------------------------------------------
    def validation_epoch_end_mean_logprob_of_n_once_tokens(self) -> float:
        # Harmonic Mean
        #return self.n_prob / self.total_logprob_of_gen_once
        # Regular Mean
        return  self.total_logprob_of_gen_once / (self.n_prob * self.n_digits)


    #-----------------------------------------------
    def check_if_best_run(self, above_rand=None) -> bool:
        save_model_flag_1, save_model_flag_2 = False, False

        epoch_digit_accuracy = self.validation_epoch_end_average_eq_tokens_by_digit(log=False)
        device = epoch_digit_accuracy.device
        self.best_run_acc['digit_accuracy'] = self.best_run_acc['digit_accuracy'].to(device=device)

        digit_where  = torch.where(epoch_digit_accuracy > self.best_run_acc['digit_accuracy'],1,0)
        
        print(f"\n{epoch_digit_accuracy} | {self.best_run_acc['digit_accuracy']} | {digit_where.sum(0)} ||  {above_rand} | {self.best_run_acc['percent_above_rand']} | {above_rand > self.best_run_acc['percent_above_rand']} ||\n")
        
        if digit_where.sum(0) >= int(self.n_digits * 0.75):
            save_model_flag_1=True

        if above_rand is not None:
            if above_rand > self.best_run_acc['percent_above_rand']:
                save_model_flag_2 = True


        if (save_model_flag_1 & save_model_flag_2):
            self.best_run_acc['digit_accuracy'] = epoch_digit_accuracy
            self.best_run_acc['percent_above_rand'] = above_rand
            return True
            
        return False

    # --------------------
    def reset_stats(self) -> None:
        del self.n_equal_tokens_per_digit, self.n_onces 
        self.n_equal_tokens_per_digit = torch.zeros((self.n_digits,), dtype=torch.float32, device=self.device)
        self.n_onces = torch.tensor(0., device=self.device)
        ######
        del self.n_prob, self.total_prob_of_labels, self.known_labels_prob_per_digit
        self.n_prob = torch.tensor(0, dtype=torch.long, device=self.device)
        self.total_prob_of_labels = torch.tensor(0., dtype=torch.float32, device=self.device)
        self.known_labels_prob_per_digit = torch.zeros((self.n_digits,), dtype=torch.float32, device=self.device)
        self.total_logprob_of_gen_once = torch.tensor(0., dtype=torch.float32, device=self.device)

        return


if __name__ == '__main__':
    pass
