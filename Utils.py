from Libs import *


class DataSet(Dataset):
    def __init__(
        self,
        file_dir: str = "Data Collection - Collection state.csv",
        numSplits: int = 128,
    ):
        self.dataHolder = []
        self.RCS = []
        self.rangeDoppler = []
        self.acoustic = []
        self.label = []
        self.b, self.a = butter(2, 0.015, btype="highpass", analog=False)

        classLabel2index = {"NO": 0, "OO": 0, "T1": 1, "T2": 1, "T3": 1}
        classCount = {0: 0, 1: np.inf}

        while not (
            classCount[0] * 0.95 < classCount[1]
            and classCount[1] < classCount[0] * 1.05
        ):
            classCount[0], classCount[1] = 0, 0
            with open(file_dir, mode="r") as file:
                for dataPoint in csv.reader(file):
                    for split in range(numSplits):
                        self.dataHolder.append(
                            (dataPoint[1], split, classLabel2index[dataPoint[1][-5:-3]])
                        )
                        classCount[classLabel2index[dataPoint[1][-5:-3]]] += 1
                        if not classLabel2index[dataPoint[1][-5:-3]]:
                            self.dataHolder.append(
                                (
                                    dataPoint[1],
                                    split,
                                    classLabel2index[dataPoint[1][-5:-3]],
                                )
                            )
                            classCount[classLabel2index[dataPoint[1][-5:-3]]] += 1
                        if (
                            not classLabel2index[dataPoint[1][-5:-3]]
                            and self.__genTrue()
                        ):
                            self.dataHolder.append(
                                (
                                    dataPoint[1],
                                    split,
                                    classLabel2index[dataPoint[1][-5:-3]],
                                )
                            )
                            classCount[classLabel2index[dataPoint[1][-5:-3]]] += 1
        # print(classCount)

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
            (#chirps, #samples) shaped radar matrix
        """

        data = np.fromfile(file_name, dtype=np.uint16)
        data = data.reshape((data.shape[0] // (128 * 256 * 8), 128, 256, 8))[
            :, :, :, [channel, channel + 4]
        ][frame]

        data = data - (data >= np.power(2, 15)) * np.power(2, 16)
        return (data[:, :, 0] + 1j * data[:, :, 1]).astype(np.complex64)

    def __readAudio(self, file_name: str) -> np.ndarray:
        """Read the audio file and resample, trim.

        Args:
            file_name: name of the audio file to read
        Returns:
            (#nSamples, ) shaped acoustic signal
        """
        droneSound, _ = librosa.load(file_name, sr=16000)
        index = np.random.randint(0, droneSound.shape[0] - 8000)
        droneSound = droneSound[index : index + 8000]
        return (droneSound - np.mean(droneSound)) / np.std(droneSound)

    def __dopplerProcess(
        self, beat_signal: np.ndarray, filter: bool = False
    ) -> np.ndarray:
        """Get a radar matrix as the input and process the range-doppler. Additioanlly, remove the zero doppler component.

        Args:
            beat_signal: (#chirps, #samples) shaped beat signal
            filter: applying the zero doppler filtering if True
        Returns:
             (#rangeBins, #dopplerBins) shaped range-doppler map
        """

        radarRange = np.rot90(np.fft.fft(beat_signal, axis=1), 1)

        if filter:
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

    def __rcsProcess(self, beat_signal: np.ndarray, filter: bool = False) -> np.ndarray:
        """Get a radar matrix as the input and process the RCS. Additioanlly, remove the zero doppler component.

        Args:
            beat_signal: (#chirps, #samples) shaped beat signal
            filter: applying the zero doppler filtering if True
        Returns:
             (#dopplerBins, ) RCS array
        """

        radarRange = np.rot90(np.fft.fft(beat_signal, axis=1), 1)

        if filter:
            radarRange = filtfilt(self.b, self.a, radarRange)

        beat_signal = np.fft.ifft(np.rot90(radarRange, 3))
        return np.mean(np.abs(beat_signal) ** 2, axis=1)

    def __len__(self):
        return len(self.dataHolder)

    def __getitem__(self, idx):
        # print(idx, self.dataHolder[idx])

        classIndex = self.dataHolder[idx][2]

        beat_signal = self.__readBin(
            f"/home/gevindu/Airforce Data/{self.dataHolder[idx][0]}.bin",
            frame=self.dataHolder[idx][1],
        )

        rangeDoppler = self.__dopplerProcess(beat_signal)
        rcs = self.__rcsProcess(beat_signal)
        acoustic = self.__readAudio(
            f"/home/gevindu/Airforce Data/{self.dataHolder[idx][0]}.wav"
        )
        return (
            torch.tensor(rangeDoppler),
            torch.tensor(rcs),
            torch.tensor(acoustic),
            torch.tensor(classIndex),
        )


if __name__ == "__main__":
    dataset = DataSet()
    train_set = DataLoader(dataset, batch_size=64, shuffle=True)

    for X1, X2, X3, y in train_set:
        print(X1.shape)
        print(X2.shape)
        print(X3.shape)
        print(y.shape)
        break
