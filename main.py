import torch
from tqdm import tqdm
from model import Transformer
from config import get_config
from loss_func import CELoss, SupConLoss, DualLoss
from data_utils import load_data
from transformers import logging, AutoTokenizer, AutoModel

from sklearn.metrics import f1_score


class Instructor:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.logger.info('> creating model {}'.format(args.model_name))
        if args.model_name == 'bert':
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            base_model = AutoModel.from_pretrained('bert-base-uncased')
        elif args.model_name == 'roberta':
            self.tokenizer = AutoTokenizer.from_pretrained('roberta-base', add_prefix_space=True)
            base_model = AutoModel.from_pretrained('roberta-base')
        else:
            raise ValueError('unknown model')
        self.model = Transformer(base_model, args.num_classes, args.method)
        self.model.to(args.device)
        if args.device.type == 'cuda':
            self.logger.info('> cuda memory allocated: {}'.format(torch.cuda.memory_allocated(args.device.index)))
        self._print_args()

    def _print_args(self):
        self.logger.info('> training arguments:')
        for arg in vars(self.args):
            self.logger.info(f">>> {arg}: {getattr(self.args, arg)}")

    def _train_original(self, dataloader, criterion, optimizer):
        train_loss, n_correct, n_train = 0, 0, 0
        self.model.train()
        for inputs, targets in tqdm(dataloader, disable=self.args.backend, ascii=' >='):
            inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
            targets = targets.to(self.args.device)
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * targets.size(0)
            n_correct += (torch.argmax(outputs['predicts'], -1) == targets).sum().item()
            n_train += targets.size(0)
        return train_loss / n_train, n_correct / n_train

    
    def _train(self, dataloader, criterion, optimizer):
        train_loss, n_correct, n_train = 0, 0, 0
        y_true, y_pred = [], []  # Lists to store true labels and predicted labels

        labels_dict = {}

        self.model.train()
        for inputs, targets in tqdm(dataloader, disable=self.args.backend, ascii=' >='):
            inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
            targets = targets.to(self.args.device)
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * targets.size(0)
            
            # Compute F1 score
            predicted_labels = torch.argmax(outputs['predicts'], -1)
            n_correct += (predicted_labels == targets).sum().item()
            n_train += targets.size(0)

            # for pred, label in zip(predicted_labels, targets):
            #     if label not in labels_dict.keys():
            #         labels_dict[label] = 1
            #     else:
            #         labels_dict[label] += 1

            
            # Append true and predicted labels for F1 score calculation
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(predicted_labels.cpu().numpy())

        # Calculate F1 score
        f1 = f1_score(y_true, y_pred, average='weighted')
        #print("Labels dict: ", labels_dict)
        
        return train_loss / n_train, n_correct / n_train, f1

    def _test_original(self, dataloader, criterion):
        test_loss, n_correct, n_test = 0, 0, 0
        self.model.eval()
        with torch.no_grad():
            for inputs, targets in tqdm(dataloader, disable=self.args.backend, ascii=' >='):
                inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
                targets = targets.to(self.args.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item() * targets.size(0)
                n_correct += (torch.argmax(outputs['predicts'], -1) == targets).sum().item()
                n_test += targets.size(0)
        return test_loss / n_test, n_correct / n_test


    def _test(self, dataloader, criterion):
        test_loss, n_correct, n_test = 0, 0, 0
        y_true, y_pred = [], []  # Lists to store true labels and predicted labels

        self.model.eval()
        with torch.no_grad():
            for inputs, targets in tqdm(dataloader, disable=self.args.backend, ascii=' >='):
                inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
                targets = targets.to(self.args.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item() * targets.size(0)
                
                # Compute F1 score
                predicted_labels = torch.argmax(outputs['predicts'], -1)
                n_correct += (predicted_labels == targets).sum().item()
                n_test += targets.size(0)
                
                # Append true and predicted labels for F1 score calculation
                y_true.extend(targets.cpu().numpy())
                y_pred.extend(predicted_labels.cpu().numpy())

        # Calculate F1 score
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        return test_loss / n_test, n_correct / n_test, f1


    def run(self):
        train_dataloader, test_dataloader = load_data(dataset=self.args.dataset,
                                                      data_dir=self.args.data_dir,
                                                      tokenizer=self.tokenizer,
                                                      train_batch_size=self.args.train_batch_size,
                                                      test_batch_size=self.args.test_batch_size,
                                                      model_name=self.args.model_name,
                                                      method=self.args.method,
                                                      workers=0)
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        if self.args.method == 'ce':
            criterion = CELoss()
        elif self.args.method == 'scl':
            criterion = SupConLoss(self.args.alpha, self.args.temp)
        elif self.args.method == 'dualcl':
            criterion = DualLoss(self.args.alpha, self.args.temp)
        else:
            raise ValueError('unknown method')
        optimizer = torch.optim.AdamW(_params, lr=self.args.lr, weight_decay=self.args.decay)
        best_loss, best_acc = 0, 0
        train_f1 = -1
        test_f1 = -1  # Testing

        for epoch in range(self.args.num_epoch):
            # train_loss, train_acc = self._train_original(train_dataloader, criterion, optimizer)
            # test_loss, test_acc = self._test_original(test_dataloader, criterion)
            
            train_loss, train_acc, train_f1 = self._train(train_dataloader, criterion, optimizer)
            test_loss, test_acc, test_f1 = self._test(test_dataloader, criterion)
            # train_loss, train_acc = self._train(train_dataloader, criterion, optimizer)
            # test_loss, test_acc = self._test(test_dataloader, criterion)
            if test_acc > best_acc or (test_acc == best_acc and test_loss < best_loss):
                best_acc, best_loss = test_acc, test_loss

            self.logger.info('{}/{} - {:.2f}%'.format(epoch + 1, self.args.num_epoch, 100 * (epoch + 1) / self.args.num_epoch))
            self.logger.info('[train] loss: {:.4f}, acc: {:.2f}, f1: {:.4f}'.format(train_loss, train_acc * 100, train_f1))
            self.logger.info('[test] loss: {:.4f}, acc: {:.2f}, f1: {:.4f}'.format(test_loss, test_acc * 100, test_f1))
        self.logger.info('best loss: {:.4f}, best acc: {:.2f}'.format(best_loss, best_acc * 100))
        self.logger.info('log saved: {}'.format(self.args.log_name))

        #    self.logger.info('{}/{} - {:.2f}%'.format(epoch+1, self.args.num_epoch, 100*(epoch+1)/self.args.num_epoch))
        #     self.logger.info('[train] loss: {:.4f}, acc: {:.2f}'.format(train_loss, train_acc*100))
        #     self.logger.info('[test] loss: {:.4f}, acc: {:.2f}'.format(test_loss, test_acc*100))
        # self.logger.info('best loss: {:.4f}, best acc: {:.2f}'.format(best_loss, best_acc*100))
        # self.logger.info('log saved: {}'.format(self.args.log_name))


if __name__ == '__main__':
    logging.set_verbosity_error()
    args, logger = get_config()
    ins = Instructor(args, logger)
    ins.run()
