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
