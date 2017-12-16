#! /bin/bash

# argument parsing
N_ARGS=1
if [ "$#" -lt ${N_ARGS} ]; then
	echo "Usage: sh extract.sh <path-to-file-list>"
	exit
fi
file_list=${1}

XMP_HEADER_TAG=xpacket
XMP_HEADER="<?xpacket begin=\"\" id=\"W5M0MpCehiHzreSzNTczkc9d\"?>"
XMP_FOOTER="<?xpacket end=\"w\"?>"
TMP_FILE=shtemp

# detect an invalid file list.
[ ! -e ${file_list} ] && {
    echo "ERROR: File <${file_list}> does not exist."
    exit
} 

function validate_xmp() {
    fn="${1}"
	xpacket_present=`head -n1 "${fn}" | grep -c ${XMP_HEADER_TAG}`
	[ "${xpacket_present}" -eq "0" ] && {
	    (echo ${XMP_HEADER}; cat "${fn}"; echo ${XMP_FOOTER};) > ${TMP_FILE}
	    mv ${TMP_FILE} "${fn}"
    }
}

while read -r fn; do
    # handle Lightroom exporting XMP files without the necessary header and footer
    validate_xmp "${fn}"
done < ${file_list}

# perform the main extraction on the XMP files
#python extract.py -f ${file_list}

