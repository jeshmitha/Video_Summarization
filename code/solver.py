import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import json
import pandas as pd
from data_loader import get_loader
from modules import EncoderBlock
from modules import FrameScoring
from modules import Generator
from modules import Critic
import os

device = torch.device('cuda:0')

original_label = torch.tensor([1.0]).to(device)
summary_label = torch.tensor([0.0]).to(device)

class Solver(torch.nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config
        self.train_loader = get_loader(config.setting, 'train', config.split_index)
        self.test_loader = get_loader(config.setting, 'test', config.split_index)
        if not os.path.exists(self.config.train_score_dir):
            os.makedirs(self.config.train_score_dir)
        if not os.path.exists(self.config.test_score_dir):
            os.makedirs(self.config.test_score_dir)
        if not os.path.exists(self.config.model_dir):
            os.makedirs(self.config.model_dir)

    def build(self):

        self.encoder=EncoderBlock(self.config.input_dim, self.config.num_heads, self.config.dim_feedforward, self.config.dropout_p).to(device)
        self.framescorer=FrameScoring(self.config.input_dim).to(device)
        self.generator = Generator(self.config.input_dim, self.config.input_dim, self.config.num_heads, self.config.dropout_p, self.config.leaky_relu_negative_slope).to(device)
        self.critic = Critic(self.config.input_dim, self.config.leaky_relu_negative_slope).to(device)
        
        self.model = nn.ModuleList([self.encoder, self.framescorer, self.generator, self.critic]).to(device)

        if self.config.mode == 'train':
            self.optimizer_ED = torch.optim.Adam(self.encoder.parameters(), lr=self.config.encoder_lr)
            self.optimizer_S = torch.optim.Adam(self.framescorer.parameters(), lr=self.config.frame_scorer_lr)
            self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.config.generator_lr)
            self.optimizer_D = torch.optim.Adam(self.critic.parameters(), lr=self.config.discriminator_lr)

    def reconstruction_loss(self, h_origin, h_sum):
        """L2 loss between original-regenerated features at cLSTM's last hidden layer"""
        return torch.norm(h_origin - h_sum, p=2)
    
    criterion = nn.MSELoss()
    def sparsity_loss(self, scores):
        """Summary-Length Regularization"""

        return torch.abs(torch.mean(scores) - self.config.regularization_factor)

    def train(self):
        for epoch_i in range(self.config.n_epochs):
            self.model.train()
            for sample_i, image_tensor in enumerate(self.train_loader):
                
                print("epoch", epoch_i, "sample", sample_i)
                self.model.train()
                image_features = image_tensor[0]

                image_features_ = Variable(image_features, requires_grad=True).to(device)

                ###################### training encoder ##############
                print("training encoder-decoder")
                encoded_image_features, graph= self.encoder(image_features_)

                generated_features=self.generator(encoded_image_features, graph).to(device)

                h_sum, c_o_fake=self.critic(generated_features)

                h_origin, c_o_real=self.critic(image_features_)
                reconstruction_loss = self.reconstruction_loss(h_origin, h_sum)

                self.optimizer_ED.zero_grad()

                reconstruction_loss.backward()
                self.optimizer_ED.step()

                del encoded_image_features, graph, generated_features, h_origin, h_sum, c_o_fake, c_o_real, reconstruction_loss

                ###################### training frame scorer ##############
                print("training frame scorer")
                encoded_image_features, graph= self.encoder(image_features_)
                scores=self.framescorer(encoded_image_features)
                s_loss=self.sparsity_loss(scores)
                
                self.optimizer_S.zero_grad()
                s_loss.backward()
                self.optimizer_S.step()

                del encoded_image_features, graph, scores, s_loss

                ###################### training discriminator ##############
                print("training discriminator")
                self.optimizer_D.zero_grad()

                h_origin, c_o_real=self.critic(image_features_)
                c_o_loss = self.criterion(c_o_real, original_label)
                c_o_loss.backward()
                
                encoded_image_features,  graph= self.encoder(image_features_)
                generated_features=self.generator(encoded_image_features, graph)
                h_sum, c_o_fake=self.critic(generated_features)

                c_f_loss = self.criterion(c_o_fake, summary_label)
                c_f_loss.backward()

                for p in self.critic.parameters():
                        p.data.clamp_(-self.config.clip, self.config.clip)
                        
                self.optimizer_D.step()

                del encoded_image_features, graph, generated_features, h_sum, h_origin, c_o_fake, c_o_real, c_o_loss, c_f_loss
                
                ###################### training generator ##############
                print("training generator")

                encoded_image_features, graph= self.encoder(image_features_)
                generated_features=self.generator(encoded_image_features, graph)
                h_sum, c_o_fake=self.critic(generated_features)

                g_loss = self.criterion(c_o_fake, original_label)

                self.optimizer_G.zero_grad()

                g_loss.backward()

                self.optimizer_G.step()

                torch.cuda.empty_cache()

            # Save parameters at checkpoint
            ckpt_path = str(self.config.model_dir) + f'/epoch-{epoch_i}.pkl'
            torch.save(self.model.state_dict(), ckpt_path)
            
            self.train_evaluate(epoch_i)
            self.test_evaluate(epoch_i)
    
    def test_evaluate(self, epoch_i):
        print("evaluating on test...")
        self.model.eval()
        out_dict = {}
        for sample_i, video_tensor in enumerate(self.test_loader):

            result = {}

            video_features = video_tensor[0]
            video_name=video_tensor[1]
            nf=video_features.shape[0]

            video_features_ = Variable(video_features).to(device)

            with torch.no_grad():
                encoded_video_features, graph= self.encoder(video_features_)
                frame_scores=self.framescorer(encoded_video_features).to(device)
                generated_video_features=self.generator(encoded_video_features, graph).to(device)
                h_sum, critic_fake=self.critic(generated_video_features)
                h_origin, critic_real=self.critic(video_features_)

                result['frame_scores'] = frame_scores.tolist()
                reconstruction_loss = self.reconstruction_loss(h_origin, h_sum)
                result['reconstruction_loss'] = reconstruction_loss.item()
                sparsity_l=self.sparsity_loss(frame_scores)
                result['sparsity_loss']=sparsity_l.item()
                gen_loss = self.criterion(critic_fake, original_label)
                result['g_loss']=gen_loss.item()
                c_o_loss = self.criterion(critic_real, original_label)
                c_f_loss = self.criterion(critic_fake, summary_label)
                result['d_o_loss']=c_o_loss.item()
                result['d_f_loss']=c_f_loss.item()
                result['critic_real']=critic_real.item()
                result['critic_fake']=critic_fake.item()

                out_dict[video_name] = result
            
            test_score_save_path = self.config.test_score_dir.joinpath(f'epoch_{epoch_i}.json')

            with open(test_score_save_path, "w+") as f:
                json.dump(out_dict, f)
    
    def train_evaluate(self, epoch_i):
        print("evaluating on train...")
        self.model.eval()
        out_dict = {}
        for sample_i, video_tensor in enumerate(self.train_loader):

            result = {}

            video_features = video_tensor[0]
            video_name=video_tensor[1]
            nf=video_features.shape[0]

            video_features_ = Variable(video_features).to(device)
            with torch.no_grad():
                encoded_video_features, graph= self.encoder(video_features_)
                frame_scores=self.framescorer(encoded_video_features).to(device)
                generated_video_features=self.generator(encoded_video_features, graph).to(device)
                h_sum, critic_fake=self.critic(generated_video_features)
                h_origin, critic_real=self.critic(video_features_)
                reconstruction_loss = self.reconstruction_loss(h_origin, h_sum)
                result['reconstruction_loss'] = reconstruction_loss.item()
                sparsity_l=self.sparsity_loss(frame_scores)
                result['sparsity_loss']=sparsity_l.item()
                gen_loss = self.criterion(critic_fake, original_label)
                result['g_loss']=gen_loss.item()
                c_o_loss = self.criterion(critic_real, original_label)
                c_f_loss = self.criterion(critic_fake, summary_label)
                result['d_o_loss']=c_o_loss.item()
                result['d_f_loss']=c_f_loss.item()
                result['critic_real']=critic_real.item()
                result['critic_fake']=critic_fake.item()

                out_dict[video_name] = result
            train_score_save_path = self.config.train_score_dir.joinpath(f'epoch_{epoch_i}.json')

            with open(train_score_save_path, "w+") as f:
                json.dump(out_dict, f)