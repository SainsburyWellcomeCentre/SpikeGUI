import QtQuick 2.5
import QtQuick.Window 2.2
import QtQuick.Controls 1.3


Window {
    id: window1
    visible: true

    width: 1200
    height: 1200

    MouseArea {
        anchors.rightMargin: 0
        anchors.bottomMargin: 0
        anchors.leftMargin: 0
        anchors.topMargin: 0
        anchors.fill: parent
        onClicked: {
        }

    }

    Text {
        id: txt1
        color: "#df3cc4"

        text: "COMPARE CLUSTERS AND CONDITIONS"
        style: Text.Raised
        font.family: "Verdana"
        font.bold: true
        styleColor: "#797979"
        horizontalAlignment: Text.AlignLeft

        anchors.top: parent.top
        anchors.horizontalCenter: parent.horizontalCenter

        anchors.horizontalCenterOffset: 0
        anchors.topMargin: 20


        font.pointSize: 32


        onTextChanged: {
            console.log("I've changed !")
        }
    }

    Row {

        id: updateReset

        width: 150
        height: 50
        anchors.left: queryElements.left
        anchors.top: queryElements.bottom


        ToolButton {
            id: updateSettings
            objectName: "updateSettings"

            width: updateReset.width
            height: updateReset.height
            anchors.top: updateReset.top
            anchors.topMargin: 10

            text: "Load Options"
            rotation: 0
            activeFocusOnPress: true
            tooltip: "Update the list in the combo boxes"

            onClicked: {
                console.log("Ooh I've been clicked");
                optionGrid1.reload();
                optionGrid2.reload();
                optionGrid3.reload();
                optionGrid4.reload();
            }
        }

    }

    Column {
        id: setConditions

        width: 150
        height: 50
        anchors.top: queryElements.top
        anchors.left: queryElements.right
        anchors.leftMargin: 10

    ToolButton {
        id: updateCondition1

        width: setConditions.width
        height: setConditions.height

        text: "Set Condition A"
        rotation: 0
        tooltip: "Sets conditions for first group"
        onClicked: queryElements.get_keys(0)
    }

    ToolButton {
        id: updateCondition2

        width: updateCondition1.width
        height: updateCondition1.height

        anchors.horizontalCenter: txt1.horizontalCenter
        anchors.top: updateCondition1.bottom
        anchors.topMargin: 10

        text: "Set Condition B"
        tooltip: "Sets conditions for comparison group"
        onClicked: queryElements.get_keys(1)
    }
    ToolButton {
        id: reset

        width: updateCondition1.width
        height: updateCondition1.height

        anchors.horizontalCenter: txt1.horizontalCenter
        anchors.top: updateCondition2.bottom
        anchors.topMargin: 10

        text: "Reset Conditions"
        tooltip: "clears dictionaries"
        onClicked: py_iface.reset_conditions()
    }

}

    ToolButton {
        id: compute

        width: updateCondition1.width
        height: updateCondition1.height

        anchors.right: parent.right
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 20
        anchors.rightMargin: 50
        anchors.top: displayStats.bottom

        text: "compute"
        tooltip: "computes"
        onClicked: displayStats.set_text(py_iface.compare())
    }


    ToolButton {
        id: display

        width: updateCondition1.width
        height: updateCondition1.height

        anchors.right: compute.left
        anchors.top: displayStats.bottom
        anchors.bottom: parent.bottom

        anchors.bottomMargin: 20

        text: "display trials"
        tooltip: "display trials"
        onClicked: {
            resultsTable1.set_text(py_iface.display_table(0));
            resultsTable2.set_text(py_iface.display_table(1));
            py_iface.generate_plots();
            analysisImage1.reload();
            analysisImage2.reload();

        }
    }


    Column{
        id: queryElements
        anchors.top: parent.top
        anchors.left: parent.left
        anchors.topMargin: 130
        anchors.leftMargin: 20
        width: 350
        height: 184
        rotation: 0
        spacing: 2

        function get_keys(dict_idx) {
            for (var i = 0; i < children.length; i++)
            {
                py_iface.update_condition_dictionary(dict_idx, children[i].get_key(), children[i].get_value(), children[i].get_comparator());

            }
        }

        OptionGrid {
            id: optionGrid1
        }

        OptionGrid {
            id: optionGrid2
        }

        OptionGrid {
            id: optionGrid3
        }

        OptionGrid {
            id: optionGrid4
        }
    }

    Column {
        id: tableDisplay
        height: parent.height / 1.5
        width: parent.width / 2
        anchors.top: updateReset.bottom
        anchors.left: parent.left
        anchors.leftMargin: 25
        anchors.topMargin: 20

        spacing: 20

        TextArea {
            id: resultsTable1
            textColor: "#e619db"
            textFormat: Text.RichText
            wrapMode: TextEdit.Wrap

            width: parent.width
            height: parent.height / 3

            function set_text(txt){
                resultsTable1.text = txt
            }
        }

        TextArea {
            id: resultsTable2
            highlightOnFocus: false
            textColor: "#e81ce8"

            width: parent.width
            height: parent.height / 3

            wrapMode: TextEdit.Wrap
            textFormat: Text.RichText

            function set_text(txt){
                resultsTable2.text = txt
            }
        }

        TextArea {
            id: log
            objectName: "log"

            width: parent.width
            height: parent.height / 6

            wrapMode: TextEdit.Wrap
            textFormat: Text.RichText

            anchors.bottom: compute.top
            anchors.bottomMargin: 100

            function set_text(txt){
                resultsTable2.text = txt
            }
        }


       }

    Column {
        id: tableDisplay2

        height: parent.height / 1.5
        width: parent.width / 2


        anchors.top: updateReset.bottom
        anchors.right: parent.right
        anchors.left: tableDisplay.right

        anchors.rightMargin: 25
        anchors.leftMargin: 20
        anchors.topMargin: 20
        anchors.bottomMargin: 0

        spacing: 20

        Image {
                id: analysisImage1

                width: parent.width
                height: parent.height / 3
                rotation: 0

                source: "image://analysisprovider1/img";
                function reload() {
                    var oldSource = source;
                    source = "";
                    source = oldSource;
                }
                sourceSize.height: height
                sourceSize.width: width
                cache: false
            }

        Image {
                id: analysisImage2

                width: parent.width
                height: parent.height / 3
                rotation: 0

                source: "image://analysisprovider2/img";
                function reload() {
                    var oldSource = source;
                    source = "";
                    source = oldSource;
                    console.log('reloading');
                }
                sourceSize.height: height
                sourceSize.width: width
                cache: false
            }

        TextArea {
            id: displayStats
            objectName: "stats"
            font.family: "Verdana"
            textColor: "#e615ce"
            width: parent.width
            font.pointSize: 10

            wrapMode: TextEdit.Wrap
            textFormat: Text.RichText


            height: parent.height / 6
            anchors.bottom: compute.top
            anchors.bottomMargin: 100


            function set_text(txt){
                displayStats.text = txt
            }
        }

    }

}
