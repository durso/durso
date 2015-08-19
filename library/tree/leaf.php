<?php
/**
 * Description of leaf
 *
 * @author durso
 */
namespace library\tree;
use library\tree\node;

class leaf extends node {
    public function hasChild(){
	return false;
    }
    public function getChildren(){
	return false;
    }
}
