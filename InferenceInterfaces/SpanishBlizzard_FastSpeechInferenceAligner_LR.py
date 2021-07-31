import os

import sounddevice
import soundfile
import torch

from InferenceInterfaces.InferenceArchitectures.InferenceFastSpeech import FastSpeech2
from InferenceInterfaces.InferenceArchitectures.InferenceMelGAN import MelGANGenerator
from PreprocessingForTTS.ProcessText import TextFrontend


class SpanishBlizzard_FastSpeechInferenceAligner(torch.nn.Module):

    def __init__(self, device="cpu", use_codeswitching=True):
        super().__init__()
        self.speaker_embedding = None
        self.device = device
        self.text2phone = TextFrontend(language="es", use_codeswitching=use_codeswitching, use_word_boundaries=False, use_explicit_eos=False)
        self.phone2mel = FastSpeech2(path_to_weights=os.path.join("Models", "FastSpeech2_BlizzardAligner_LR", "best.pt"),
                                     idim=166, odim=80, spk_embed_dim=None, reduction_factor=1).to(torch.device(device))
        self.mel2wav = MelGANGenerator(path_to_weights=os.path.join("Models", "MelGAN_Blizzard", "best.pt")).to(torch.device(device))
        self.phone2mel.eval()
        self.mel2wav.eval()
        self.to(torch.device(device))

    def forward(self, text, view=False):
        with torch.no_grad():
            phones = self.text2phone.string_to_tensor(text).squeeze(0).long().to(torch.device(self.device))
            mel, durations, pitch, energy = self.phone2mel(phones, speaker_embedding=self.speaker_embedding, return_duration_pitch_energy=True)
            mel = mel.transpose(0, 1)
            wave = self.mel2wav(mel.unsqueeze(0)).squeeze(0).squeeze(0)
        if view and len(text) < 40:
            import matplotlib.pyplot as plt
            import librosa.display as lbd
            from Utility.utils import cumsum_durations
            fig, ax = plt.subplots(nrows=2, ncols=1)
            ax[0].plot(wave.cpu().numpy())
            lbd.specshow(mel.cpu().numpy(),
                         ax=ax[1],
                         sr=16000,
                         cmap='GnBu',
                         y_axis='mel',
                         x_axis=None,
                         hop_length=256)
            ax[0].yaxis.set_visible(False)
            ax[1].yaxis.set_visible(False)
            duration_splits, label_positions = cumsum_durations(durations.cpu().numpy())
            ax[1].set_xticks(duration_splits, minor=True)
            ax[1].xaxis.grid(True, which='minor', color='black')
            ax[1].set_xticks(label_positions, minor=False)
            ax[1].set_xticklabels(self.text2phone.string_to_tensor(text=text, return_string=True))
            ax[0].set_title(text)
            plt.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=.9, wspace=0.0, hspace=0.0)
            plt.savefig("example.pdf")
            plt.show()
        elif view:
            import matplotlib.pyplot as plt
            import librosa.display as lbd
            fig, ax = plt.subplots(nrows=2, ncols=1)
            ax[0].plot(wave.cpu().numpy())
            lbd.specshow(mel.cpu().numpy(), ax=ax[1], sr=16000, cmap='GnBu', y_axis='mel', x_axis='time',
                         hop_length=256)
            plt.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=.9, wspace=0.0, hspace=0.0)
            ax[0].set(title=self.text2phone.string_to_tensor(text=text, return_string=True))
            ax[0].label_outer()
            plt.show()

        return wave

    def read_to_file(self, text_list, file_location, silent=False):
        """
        :param silent: Whether to be verbose about the process
        :param text_list: A list of strings to be read
        :param file_location: The path and name of the file it should be saved to
        """
        wav = None
        silence = torch.zeros([8000])
        for text in text_list:
            if text.strip() != "":
                if not silent:
                    print("Now synthesizing: {}".format(text))
                if wav is None:
                    wav = self(text).cpu()
                    wav = torch.cat((wav, silence), 0)
                else:
                    wav = torch.cat((wav, self(text).cpu()), 0)
                    wav = torch.cat((wav, silence), 0)
        soundfile.write(file=file_location, data=wav.cpu().numpy(), samplerate=16000)

    def read_aloud(self, text, view=False, blocking=False):
        if text.strip() == "":
            return
        wav = self(text, view).cpu()
        wav = torch.cat((wav, torch.zeros([8000])), 0)
        if not blocking:
            sounddevice.play(wav.numpy(), samplerate=16000)
        else:
            sounddevice.play(torch.cat((wav, torch.zeros([12000])), 0).numpy(), samplerate=16000)
            sounddevice.wait()
