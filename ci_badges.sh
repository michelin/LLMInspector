#!/bin/bash
pip install anybadge
mkdir -p public/quality/

exit_status=false
args_list=()
for arg in $*
do
    if [ "$arg" == "--exit-status" ]; then
        exit_status=true
    else
        args_list+=($arg)
    fi
done

# Black badge
black . --check --diff > .blackout 2>&1
black_status=$?
if [[ $black_status -eq 0 ]]; then
    BLACK_RESULT=passing
else
    BLACK_RESULT=failing
fi
anybadge -l black -v $BLACK_RESULT -f ./public/quality/black.svg passing=green failing=red

# Isort badge
isort -c --diff . > .isortout 2>&1
isort_status=$?
if [[ $isort_status -eq 0 ]]; then
    ISORT_RESULT=passing
else
    ISORT_RESULT=failing
fi
anybadge -l isort -v $ISORT_RESULT -f ./public/quality/isort.svg passing=green failing=red

# Pylint badge
pylint "${args_list[@]}" -f text --exit-zero > .pylintout 2> .pylinterror
anybadge -l pylint -v $(cat .pylintout | tail -n2 | awk '{print $7}' | cut -d"/" -f1) -f ./public/quality/pylint.svg 5=red 8=orange 9=yellow 9.5=yellowgreen 10=green

# Building htmls
echo "<pre>" > ./public/quality/black.html
cat .blackout >> ./public/quality/black.html
echo "</pre>" >> ./public/quality/black.html

echo "<pre>" > ./public/quality/isort.html
cat .isortout >> ./public/quality/isort.html
echo "</pre>" >> ./public/quality/isort.html

echo "<pre>" > ./public/quality/pylint.html
cat .pylinterror >> ./public/quality/pylint.html
cat .pylintout >> ./public/quality/pylint.html
echo "</pre>" >> ./public/quality/pylint.html

# print quality tools output
cat .blackout
cat .isortout
cat .pylinterror
cat .pylintout

if $exit_status; then
    if [ "$isort_status" -ne "0" ] || [ "$black_status" -ne "0" ]; then
        exit -1
    fi
fi
