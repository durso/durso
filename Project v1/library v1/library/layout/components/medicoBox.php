<?php

/**
 * Description of resultBox
 *
 * @author durso
 */
namespace library\layout\components;
use library\layout\components\component;
use library\layout\elements\element;
use app\model\file;
 

class medicoBox extends component{
    protected $review;
    public function __construct($row,$tag = "div",$class = array("container")) {
        $path = VIEW_PATH.DS."templates/medico.php"; 
        $this->value = include($path);
        $this->tag = $tag;
        $this->closeTag = true;
        $this->attributes["class"] = $class;
    }
    public function addReview($rows){
        $path = VIEW_PATH.DS."templates/review.php"; 
        $this->value .= include($path);
    }
   
    

    
}