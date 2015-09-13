<?php

/**
 * Description of gChart (Google Chart)
 *
 * @author durso
 */

namespace library\dom\chart;
use library\dom\elements\components\elementFactory;
use app\model\file;

class gChart {
    private static $script = null;
    private $options = array();
    private $type;
    private $rows;
    private $columns;
    private $functionData;
    private $name;
    private $n;
    private $id;
    private static $js = null;
    private static $counter = 0;
    private static $output;
    
    public function __construct($name, $type, $id){
        $this->type = $type;
        if(is_null(self::$js)){
            self::$js = file::read(__DIR__.DS."gChartTemplate.txt");
        }
        if(is_null(self::$script)){
            self::$script = elementFactory::createByTag("script");
            self::$script->attr("type","text/javascript");    
        }
        $this->name = $name;
        $this->n = ++self::$counter;
        $this->functionData = file::read(__DIR__.DS."gChartFunctionTemplate.txt");
        $this->id = $id;
    }
    
    public function addColumn($value, $dataType = 'number'){
        $this->columns[$value] = $dataType;
    }
    
    public function addRows(array $values, $typeCasting = false){
        if($typeCasting){
            foreach($values as $key => $value){
                if(is_numeric($value) && is_string($value)){
                    $values[$key] = (strpos($value,".") === false) ? ((int) $value): ((float) $value);
                } 
            }
        }
        $this->rows[] = $values;
    }
    
    public function readCSV($file){
        $csv = file::readCSV(FILES_PATH.DS.$file);
        for($i = 0; $i < count($csv); $i++){
            if($i == 0){
                for($j = 0; $j < count($csv[$i]); $j++){
                    $dataType = is_numeric($csv[$i+1][$j]) ? 'number' : 'string';
                    $this->addColumn($csv[$i][$j], $dataType);
                }
                continue;
            }
            $this->addRows($csv[$i],true);
        }
    }

    
    

    public function addOption($name,$value){
        $this->options[$name] = $value;
    }
    public function bgColor($color){
        $this->options["backgroundColor"] = $color;
    }
    public function colors(array $colors){
        $this->options["colors"] = $colors;
    }
    public function textColor($color){
        $this->options["titleTextStyle"]["color"] = $color;
    }
    public function fontSize($size){
        $this->options["fontSize"] = $size;
    }
    public function fontName($font){
        $this->options["fontName"] = $font;
    }
    public function hGridlinesColor($color){
        $this->options["hAxis"]["gridlines"]["color"] = $color;
    }
    public function hBaseLineColor($color){
        $this->options["hAxis"]["baselineColor"] = $color;
    }
    public function hRemoveLabel(){
        $this->options["hAxis"]["textPosition"] = 'none';
    }
    public function vGridlinesColor($color){
        $this->options["vAxis"]["gridlines"]["color"] = $color;
    }
    public function vBaseLineColor($color){
        $this->options["vAxis"]["baselineColor"] = $color;
    }
    public function vRemoveLabel(){
        $this->options["vAxis"]["textPosition"] = 'none';
    }
    public function removeLegend(){
        $this->options["legend"] = 'none';
    }
    public function removeGrids(){
        $this->bgColor('none');
        $this->colors(array('white'));
        $this->fontSize(16);
        $this->hGridlinesColor('none');
        $this->hBaseLineColor('none');
        $this->vGridlinesColor('none');
        $this->vBaseLineColor('none');
        $this->removeLegend();
        $this->vRemoveLabel();
        $this->hRemoveLabel();
    }
    public function prepare(){
        $options = json_encode($this->options);
        $columns = "";
        foreach($this->columns as $key => $value){
            $columns .= $this->name."."."addColumn('".$value."','".$key."');";
        }
        $rows = json_encode($this->rows);

        $search = array("@name@","@column@","@rows@","@options@","@i@","@type@","@id@");
        $replace = array($this->name,$columns,$rows,$options,$this->n,$this->type,$this->id);
        self::$output .= str_replace($search,$replace,$this->functionData);
    }

    public static function save(){

        $js = str_replace("@function@", self::$output, self::$js);
        self::$script->addComponent(elementFactory::createText($js));
        return self::$script;
    }
    

}
