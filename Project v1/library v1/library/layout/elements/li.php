<?php

/**
 * Description of li
 *
 * @author durso
 */
namespace library\layout\elements;
use library\layout\elements\element;

class li extends element{

    public function __construct($className = array()) {
        $this->attributes["class"] = $className;
        $this->tag = "li";
        $this->closeTag = true;
    }


}