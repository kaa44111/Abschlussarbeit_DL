# Evaluierung von Einsatzmöglichkeiten einer Deep Learning Applikation zur Detektion gelabelter Merkmale in Bildern

Dieses Repository enthält den Code und die Dokumentation zu meiner Bachelorarbeit mit dem Titel: 
„Evaluierung von Einsatzmöglichkeiten einer Deep Learning Applikation zur Detektion gelabelter Merkmale in Bildern“.

## Themenschwerpunkte
- **Einarbeitung in den Stand der Technik und Literaturrecherche**: Untersuchung aktueller Methoden und Technologien im Bereich der Bildsegmentierung und Deep Learning.
- **Präparierung von Bilddaten**: Vorbereitung der Bilddaten durch Labeln und Datenaugmentation mithilfe verfügbarer Bildverarbeitungssoftware.
- **Auswahl geeigneter Softwarebibliotheken zur Modellbildung**: Identifikation und Nutzung geeigneter Bibliotheken und Frameworks für die Modellierung.
- **Validierung der Auswahl und Bewertung der Ergebnisse**: Bewertung der Modellleistung anhand verschiedener Metriken und Validierung der Ergebnisse.

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


### test_models
Überprüft die in `results` gespeicherten Ergebnisse.
- `test_different_models.py`: Testet gleichzeitig alle Ergebnisse aus den 3 Modellvariationen und speichert die Ergebnisse.

### train
Enthält Skripte und Methoden zum Training der Modelle.
- `train_compare.py`: Hauptskript zum Training der Modelle. Enthält Funktionen zur Anpassung von Hyperparametern, Speicherung von Modellen und Überwachung des Trainingsfortschritts.
- `results`: Unterordner, der die Ergebnisse der antrainierten Modelle speichert. (einschließlich Modellgewichte, Trainingsprotokolle und Visualisierungen der Segmentierungsergebnisse.)*
- 
### utils
Enthält verschiedene Hilfsfunktionen und Tools, die im Projekt verwendet werden.

## Installation
1. Klone das Repository:
   ```bash
   git clone https://github.com/kaa44111/Abschlussarbeit_DL.git
2. Navigiere in das Repository-Verzeichnis:
   cd Abschlussarbeit_DL
3. Installiere die benötigten Pakete:
   pip install -r requirements.txt

## Datasets
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
