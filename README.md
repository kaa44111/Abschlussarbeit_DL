# Deep Learning Image Segmentation

Dieses Repository enthält verschiedene Modelle und Methoden zur Bildsegmentierung mit Deep Learning. Es umfasst Datensätze, Modelle, Vorverarbeitungsmethoden und Testskripte.

## Inhaltsverzeichnis
- [Struktur](#struktur)
- [Installation](#installation)
- [Datensätze](#datensätze)
- [Modelle](#modelle)
- [Vorverarbeitung](#vorverarbeitung)
- [Training](#training)
- [Ergebnisse](#ergebnisse)
- [Testen der Modelle](#testen-der-modelle)
- [Zukünftige Arbeiten](#zukünftige-arbeiten)
- [Autoren](#autoren)

## Struktur

### data
Beinhaltet die Originalbilder, die zum Training benötigt werden.
- `circle_data`: 300 Bilder, 300 Masken
- `geometry_shapes`: 300 Bilder, 1800 Masken
- `Ölflecken`: 28 Bilder, 28 Masken
- `RetinaVessel`: 20 Bilder, 20 Masken
- `WireCheck`: Noch nicht spezifiziert

### data_modified
Beinhaltet vorbearbeitete Bilder, die im Modul `prepare` erstellt wurden (z.B. Patches aus größeren Bildern oder Bilder, die gebinnt wurden).
- `RetinaVessel`: Bilder aus `RetinaVessel` werden in kleineren Patches betrachtet.

### datasets
- `MultipleFeature.py`: CustomDataset für Bilder mit mehreren Masken (Features), z.B. `geometry_shapes`.
- `OneFeature.py`: CustomDataset für Bilder mit nur einer Maske.

### models
- `UNet.py`: Definition des originalen UNet-Modells.
- `UNetBatchNorm.py`: Variation des UNet-Modells mit der Schicht `nn.BatchNorm2d()`.
- `UNetNoMaxPool.py`: Variation des UNet-Modells ohne die Schicht `nn.MaxPool2d()`.

### prepare
Verschiedene Methoden zur Vorbereitung der Bilder.
- `prepare_patches.py`: Methoden, die die Bilder in kleinere Patches aufteilen und im Ordner `modified/'name_des_original_Ordners'` speichern.
- `prepare_binning.py`: Methoden, die die Bilder binned und im Ordner `modified/'name_des_original_Ordners'` speichern.
- **Zukünftig:** `prepare_both.py`: Führt zuerst das Binning und dann das Aufteilen in Patches durch.

### results
Speichert die verschiedenen Ergebnisse der antrainierten Modelle.

### test_models
Überprüft die in `results` gespeicherten Ergebnisse.
- `test_different_models.py`: Testet gleichzeitig alle Ergebnisse aus den 3 Modellvariationen und speichert die Ergebnisse.

## Installation

## Datensätze
Die Originalbilder befinden sich im Ordner data. Modifizierte Bilder werden im Ordner data_modified gespeichert. Verwende die Skripte im Ordner prepare zur Vorverarbeitung der Bilder.

## Modelle
Die Modelldefinitionen befinden sich im Ordner models. Drei Variationen des UNet-Modells sind verfügbar:

Original UNet (UNet.py)
UNet mit Batch Normalization (UNetBatchNorm.py)
UNet ohne MaxPooling (UNetNoMaxPool.py)

## Vorverarbeitung
Verwende die Skripte im Ordner prepare zur Vorverarbeitung der Bilder:

prepare_patches.py: Teilt Bilder in kleinere Patches auf.
prepare_binning.py: Führt Binning auf den Bildern durch.
Zukünftig: prepare_both.py: Führt zuerst das Binning und dann das Aufteilen in Patches durch.
Training
Verwende das Skript train_model.py im Ordner training, um die Modelle mit den vorbereiteten Datensätzen zu trainieren. Passe die Hyperparameter nach Bedarf an. Verwende lr_scheduler.py zur Anpassung der Lernrate während des Trainings.

## Ergebnisse
Die Ergebnisse des Modelltrainings werden im Ordner results gespeichert. Dies umfasst Modellgewichte, Trainingsprotokolle und Visualisierungen der Segmentierungsergebnisse.

## Testen der Modelle
Verwende das Skript test_different_models.py im Ordner test_models, um die Ergebnisse der verschiedenen Modellvariationen zu testen und zu vergleichen.

## Zukünftige Arbeiten
Hinzufügen eines Skripts prepare_both.py, das zuerst Binning durchführt und dann die Bilder in Patches aufteilt.
Erweiterung des WireCheck-Datensatzes.
## Autoren
Dieses Projekt wurde von Amina entwickelt. Weitere Beiträge sind willkommen!
