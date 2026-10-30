@echo off
rem Rebuild both corpora with the current pipeline, then train the
rem morpheme-aware model and the plain baseline, one after the other.
rem Run from the repository root, inside the activated environment:
rem     scripts\train_all.bat
rem Needs corpus\am.txt (CC-100) and corpus\GPAC.txt. Each step writes a
rem log in logs\; the run stops at the first step that fails.

set PYTHONUTF8=1
if not exist logs mkdir logs
if not exist corpus\gpac mkdir corpus\gpac

echo [%time%] 1/4 CC-100 corpus
python -u -m amharic_morph.corpus_runner corpus\am.txt --output corpus\corpus.txt --metadata corpus\metadata.jsonl --statistics corpus\statistics.json --drop-tigrinya > logs\1_cc100.log 2>&1
if errorlevel 1 goto failed

echo [%time%] 2/4 GPAC corpus
python -u -m amharic_morph.corpus_runner corpus\GPAC.txt --output corpus\gpac\corpus.txt --metadata corpus\gpac\metadata.jsonl --statistics corpus\gpac\statistics.json --source gpac --drop-tigrinya --dedup-against corpus\corpus.txt > logs\2_gpac.log 2>&1
if errorlevel 1 goto failed

copy /b corpus\corpus.txt + corpus\gpac\corpus.txt corpus\combined.txt > nul

echo [%time%] 3/4 morpheme-aware model
python -u scripts\train_bpe.py corpus\combined.txt corpus\bpe_full --vocab 16000 --threads 8 > logs\3_morph.log 2>&1
if errorlevel 1 goto failed

echo [%time%] 4/4 plain baseline
python -u scripts\train_bpe.py corpus\combined.txt corpus\bpe_full --vocab 16000 --threads 8 --plain > logs\4_plain.log 2>&1
if errorlevel 1 goto failed

echo [%time%] ALL DONE > logs\DONE.txt
echo [%time%] all done
goto :eof

:failed
echo [%time%] FAILED, see the last log in logs\ > logs\FAILED.txt
echo [%time%] failed, see the last log in logs\
