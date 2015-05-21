<?php

/**
 * Description of layout
 *
 * @author durso
 */
namespace library\layout;
use library\layout\elements\element;

class layout extends element{
    private $container;
    public function __construct(){
        $this->tag = "body";
        $this->closeTag = true;
    }
}
