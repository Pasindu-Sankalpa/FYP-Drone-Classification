from Libs import *

class MatchAudio:
    def __init__(
        self,
        sampling_rate=16000,
        base_audio_file="",
        cut_off=2500, #2000
        filter_order=4,
        coeff_main=0.5,
        coeff_base=1,
    ) -> None:
        self.sampling_rate = sampling_rate
        self.cut_off = cut_off
        self.order = filter_order
        self.coeff_main = coeff_main
        self.coeff_base = coeff_base
        self.low_pass_data = self.__low_pass(librosa.load(base_audio_file, sr=self.sampling_rate)[0])

    def __low_pass(self, audio):
        b, a = signal.butter(
            self.order, self.cut_off / (self.sampling_rate * 0.5), "lowpass", False
        )
        return signal.filtfilt(b, a, audio)

    def __high_pass(self, audio):
        b, a = signal.butter(
            self.order, self.cut_off / (self.sampling_rate * 0.5), "highpass", False
        )
        return signal.filtfilt(b, a, audio)

    def __make_mix_signal(self, main, low_passed):
        len_main = len(main)
        len_low_passed = len(low_passed)
        if len_main == len_low_passed:
            return low_passed
        elif len_main < len_low_passed:
            return low_passed[:len_main]
        else:
            mix = np.zeros_like(main)
            for i in range(len_main // len_low_passed):
                mix[i * len_low_passed : (i + 1) * len_low_passed] = low_passed
            mix[(i + 1) * len_low_passed :] = low_passed[
                : len_main - (i + 1) * len_low_passed
            ]
            return mix

    def match(self, audio_name):
        main_data, _ = librosa.load(audio_name, sr=self.sampling_rate)
        
        high_pass_data = self.__high_pass(main_data)

        low_pass_data_reshape = self.__make_mix_signal(
            high_pass_data, self.low_pass_data
        )

        return (
            self.coeff_main * high_pass_data
            + self.coeff_base * low_pass_data_reshape
        )

class TestDataSet(Dataset):
    def __init__(
        self,
        fileName: os.PathLike = "",
        datasetDir: os.PathLike = "D:\FYP\Drone Classification\Pipeline\GUI_Combined\data",
        numSplits: int = 128,
        filter: bool = False,
    ):
        """Class to access the dataset collected from airforce.

        Args:
            fileName: file to be tested
            datasetDir: folder path which contains the dataset files
            numSplits: number of samples extract from a single file
            filter: apply zerp doppler filter to the radar at doppler and RCS processing
        """
        self.datasetDir = datasetDir
        self.filter = filter
        self.b, self.a = butter(2, 0.015, btype="highpass", analog=False)
        self.matcher = MatchAudio(base_audio_file = r"D:\FYP\Drone Classification\Pipeline\GUI_Combined\model\20240313_NO_01.wav")

        self.dataHolder = []

        for split in range(numSplits):
            # adding the datapoint to araay
            self.dataHolder.append(
                (fileName, split)
            )

    def __genTrue(self, trueProb: float = 0.4) -> bool:
        """Return True with a given probability, False otherwise. Used in upsampling the dataset.

        Args:
            trueProb: expected true probability

        Returns:
            Generated bool value
        """
        if random.randint(1, 100) <= trueProb * 100:
            return True
        else:
            return False

    def __readBin(self, file_name: str, channel: int = 0, frame: int = 0) -> np.ndarray:
        """Read the radar signal from binary file and convert to necessary format for further processing.

        Args:
            file_name: name of the binary file to read
            channel: RX channel
            frame: frame number

        Returns:
            (#chirps, #samples) shaped radar matrix at the index "frame"
        """

        data = np.fromfile(file_name, dtype=np.uint16)
        data = data.reshape((data.shape[0] // (128 * 256 * 8), 128, 256, 8))[
            :, :, :, [channel, channel + 4]
        ][frame]

        data = data - (data >= np.power(2, 15)) * np.power(2, 16)
        return (data[:, :, 0] + 1j * data[:, :, 1]).astype(np.complex64)

    def __readAudio(self, file_name: str, frame) -> np.ndarray:
        """Read the audio file and resample, trim.

        Args:
            file_name: name of the audio file to read
        Returns:
            (#nSamples, ) shaped acoustic signal
        """
        frameRate = 25
        startTime = int(frame / frameRate * 1000)

        droneSound = self.matcher.match(file_name)
        # droneSound, _ = librosa.load(file_name, sr=16000)

        if startTime + 8000 <= droneSound.shape[0]:
            droneSound = droneSound[startTime : startTime + 8000]
        else:
            index = np.random.randint(0, droneSound.shape[0] - 8000)
            droneSound = droneSound[index : index + 8000]
        return (droneSound - np.mean(droneSound)) / np.std(droneSound)

    def __dopplerProcess(self, beat_signal: np.ndarray) -> np.ndarray:
        """Get a radar matrix as the input and process the range-doppler. Additioanlly, remove the zero doppler component.

        Args:
            beat_signal: (#chirps, #samples) shaped beat signal
        Returns:
             (#rangeBins, #dopplerBins) shaped range-doppler map
        """

        radarRange = np.rot90(np.fft.fft(beat_signal, axis=1), 1)

        if self.filter:
            radarRange = filtfilt(self.b, self.a, radarRange)

        doppler = np.abs(np.fft.fft(radarRange))

        doppler_map = np.empty_like(doppler)
        doppler_map[:, :64], doppler_map[:, 64:] = (
            doppler[:, 64:],
            doppler[:, :64],
        )

        return np.uint8(
            (doppler_map - np.min(doppler_map))
            / (np.max(doppler_map) - np.min(doppler_map))
            * 255
        )

    def __rcsProcess(self, beat_signal: np.ndarray) -> np.ndarray:
        """Get a radar matrix as the input and process the RCS. Additioanlly, remove the zero doppler component.

        Args:
            beat_signal: (#chirps, #samples) shaped beat signal
        Returns:
             (#dopplerBins, ) RCS array
        """

        radarRange = np.rot90(np.fft.fft(beat_signal, axis=1), 1)

        if self.filter:
            radarRange = filtfilt(self.b, self.a, radarRange)

        beat_signal = np.fft.ifft(np.rot90(radarRange, 3))
        return np.mean(np.abs(beat_signal) ** 2, axis=1)

    def __len__(self):
        return len(self.dataHolder)

    def __getitem__(self, idx):
        beat_signal = self.__readBin(
            f"{self.datasetDir}/{self.dataHolder[idx][0]}.bin",
            frame=self.dataHolder[idx][1],
        )

        return (
            torch.tensor(self.__dopplerProcess(beat_signal), dtype=torch.float),
            torch.tensor(self.__rcsProcess(beat_signal), dtype=torch.float),
            torch.tensor(
                self.__readAudio(
                    f"{self.datasetDir}/{self.dataHolder[idx][0]}.wav",
                    frame=self.dataHolder[idx][1],
                ),
                dtype=torch.float,
            ),
        )

if __name__ == "__main__":
    pass
