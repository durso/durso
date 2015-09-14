<?php

namespace library\dom\structures;
use library\dom\structures\components;
use library\dom\elements\components\text;
use library\dom\elements\components\block;
use library\dom\object;
use app\model\file;


class sTable extends components{
    protected $tbody;
    protected $thead;
    protected $tfoot;


    public function __construct(){
        parent::__construct("table");
        $this->tbody = null;
        $this->thead = null;
        $this->tfoot = null;
    }

    public function create(array $values,$header = false){
        $this->group("tbody");
        for($i = 0; $i < count($values); $i++){
            if($header){
                if($i == 0){
                    $this->group("thead");
                    $this->addHeader($values[$i]);
                    continue;
                }
            }
            $row = $this->addRow($this->tbody);
            for($j = 0; $j < count($values[$i]); $j++){
                $col = $this->addCell('td',$values[$i][$j],$row);
            }
        }
    }

    public function addHeader(array $values){
        $this->add($values,"thead","th");
    }

    public function addFooter(array $values){
        $this->add($values,"tfoot","td");
    }
    
    /*
     * This function is used to create a group element and
     * then add children cells to it
     * @param array $values the values to be added to each cell
     * @param string $dest the html tag for the group (tbody,thead or tfoot)
     * @param string $cell the html tag for the cell
     * @return void
     */
    protected function add(array $values,$dest,$cell){
        $this->group($dest);
        $row = $this->addRow($this->$dest);
        foreach($values as $value){
           $this->addCell($cell,$value,$row);
        }
    }
    
    /*
     * This function is used to add a group element (tbody,thead or tfoot) to the sTable
     * @param string $tag the html tag (tbody,thead or tfoot)
     * @return void
     */
    public function group($tag){
        if(property_exists($this,$tag)){
            if(is_null($this->$tag)){
                $this->$tag = new block($tag);
                //adds the group element to the table
                $this->addComponent($this->$tag);
            }
        }
    }
    
    /*
     * This function is used to add a table row (tr)
     * @param mixed $dest the destination of the table row. It can be a string or a object
     * @return block
     */
    public function addRow($dest = false){
        $tr = new block("tr");
        if(!$dest){
            $this->addComponent($tr);
        } else {
            if($dest instanceof object){
                $dest->addComponent($tr);
            } else {
                switch($dest){
                 case 'tbody': $this->tbody->addComponent($tr);
                     break;
                 case 'thead': $this->thead->addComponent($tr);
                     break;
                 case 'tfoot': $this->tfoot->addComponent($tr);
                     break;
                }
            }
        }
        return $tr;
    }
    
    
    /*
     * This function is used to add a cell to the table
     * @param string $tag the html tag
     * @param string $value the value of the cell
     * @param object $dest the parent node for the cell
     * @return void
     */
    public function addCell($tag,$value,$dest){
        $cell = new block($tag);
        $text = new text($value);
        $cell->addComponent($text);
        $dest->addComponent($cell);
    }
    
    /*
     * Read a CSV file and adds it to the table
     * @param string $file the full path to the file
     * @param boolean $header if the CSV file has headers
     * @return void
     */
    public function readCSV($file,$header = false){
        $csv = file::readCSV($file);
        $this->create($csv,$header);
    }
    
}