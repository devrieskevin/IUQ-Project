import sys
import re

if __name__ == "__main__":
    nodelist = sys.argv[1]

    normal_nodes = re.findall(r"r[0-9]+n[0-9]+",nodelist)
    array_nodes = re.findall(r"r[0-9]+n\[[0-9-,]+\]",nodelist)

    for nodes in array_nodes:
        first,second = nodes.split("[")
        second = second[:-1]

        slices = second.split(",")
        for elem in slices:
            if "-" in elem:
                imin,imax = elem.split("-")
                res = [first + str(n) for n in range(int(imin),int(imax)+1)]
                normal_nodes = normal_nodes + res
            else:
                normal_nodes.append(first + elem)

    print(*normal_nodes,sep=" ")
