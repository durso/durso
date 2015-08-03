<?php

namespace library\dom\structures;
use library\dom\structures\components;
use library\dom\elements\components\text;
use library\dom\elements\components\block;
use library\dom\object;
use app\model\file;


class sTable extends components{
    protected $tbody =  null;
    protected $thead = null;
    protected $tfoot = null;


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
    
    protected function add(array $values,$dest,$cell){
        $this->group($dest);
        $row = $this->addRow($this->$dest);
        foreach($values as $value){
           $this->addCell($cell,$value,$row);
        }
    }

    public function group($tag){
        if(property_exists($this,$tag)){
            if(is_null($this->$tag)){
                $this->$tag = new block($tag);
            }
        }
    }
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
            $this->tracker($tr);
        }
        return $tr;
    }
    public function readCSV($file,$header = false){
        $csv = file::readCSV($file);
        $this->create($csv,$header);
    }
    public function addCell($tag,$value,$dest){
        $cell = new block($tag);
        $text = new text($value);
        $cell->addComponent($text);
        $dest->addComponent($cell);
        $this->tracker($cell);
    }
    
    public function save(){
        if(!is_null($this->thead)){
            $this->root->addComponent($this->thead);
        }
        if(!is_null($this->tbody)){
            $this->root->addComponent($this->tbody);
        }
        if(!is_null($this->tfoot)){
            $this->root->addComponent($this->tfoot);
        }
        return $this->root;
    }
}