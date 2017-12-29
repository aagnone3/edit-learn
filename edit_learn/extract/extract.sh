#! /bin/bash

# argument parsing
N_ARGS=2
if [ "$#" -lt ${N_ARGS} ]; then
	echo "Usage: sh extract.sh <path-to-file-list> <output-path>"
	exit
fi
file_list=${1}
out_fn=${2}

XMP_HEADER_TAG=xpacket
XMP_HEADER="<?xpacket begin=\"\" id=\"W5M0MpCehiHzreSzNTczkc9d\"?>"
XMP_FOOTER="<?xpacket end=\"w\"?>"
TMP_FILE=shtemp
EXTRACT_LABELS_SCRIPT=extract.py
XMP_LIST_FN="xmps.lst"
NEF_LIST_FN="nefs.lst"

# detect an invalid file list.
[ ! -e ${file_list} ] && {
    echo "ERROR: File <${file_list}> does not exist."
    exit
}

validate_xmp() {
    # handle Lightroom exporting XMP files without the necessary header and footer
    fn="${1}"
    [ -e "${fn}" ] && {
        xpacket_present=$(head -n1 "${fn}" | grep -c ${XMP_HEADER_TAG})
        [ "${xpacket_present}" -eq "0" ] && {
            (echo ${XMP_HEADER}; cat "${fn}"; echo ${XMP_FOOTER};) > ${TMP_FILE}
            mv ${TMP_FILE} "${fn}"
        }
    } || {
        return 1
    }
}

grep -i xmp ${file_list} > ${XMP_LIST_FN}
n_xmp=$(cat ${XMP_LIST_FN} | wc -l)
[ "${n_xmp}" -eq "0" ] && {
    echo "ERROR: No XMP files found in ${XMP_LIST_FN}."
    exit
} || {
    echo "# XMP files found: ${n_xmp}"
}
grep -i nef ${file_list} > ${NEF_LIST_FN}
n_nef=$(cat ${NEF_LIST_FN} | wc -l)
[ "${n_nef}" -eq "0" ] && {
    echo "ERROR: No NEF files found in ${NEF_LIST_FN}."
    exit
} || {
    echo "# NEF files found: ${n_nef}"
}

# validate the XMP files
while read -r fn; do
    validate_xmp "${fn}"
    [ "$?" -ne "0" ] && {
        echo "EROR: File "${fn}". does not exist."
    }
done < ${XMP_LIST_FN}

# perform label extraction on the XMP files
python ${EXTRACT_LABELS_SCRIPT} -i ${NEF_LIST_FN} -x ${XMP_LIST_FN} -o ${out_fn}
rc=$?
[ "${rc}" -ne 0 ] && {
    echo "ERROR: Dataset extraction failed with exit code ${rc}."
}

